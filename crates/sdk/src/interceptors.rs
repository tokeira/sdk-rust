//! User-definable interceptors are defined in this module

use crate::{
    TimerResult, Worker, WorkflowContextView, WorkflowResult,
    workflow_context::{CancellableFuture, TimerOptions},
    workflows::WorkflowError,
};
use anyhow::bail;
use futures_util::{FutureExt, future::LocalBoxFuture};
use std::{
    any::TypeId,
    collections::HashMap,
    fmt,
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{Arc, OnceLock},
    task::{Context, Poll},
};
use temporalio_common::{
    data_converters::TemporalSerializable,
    protos::{
        coresdk::{
            workflow_activation::{WorkflowActivation, remove_from_cache::EvictionReason},
            workflow_completion::WorkflowActivationCompletion,
        },
        temporal::api::common::v1::{Payload, Payloads},
    },
};

/// Implementors can intercept certain actions that happen within the Worker.
///
/// Advanced usage only.
#[async_trait::async_trait(?Send)]
pub trait WorkerInterceptor {
    /// Called every time a workflow activation completes (just before sending the completion to
    /// core).
    async fn on_workflow_activation_completion(&self, _completion: &WorkflowActivationCompletion) {}
    /// Called after the worker has initiated shutdown and the workflow/activity polling loops
    /// have exited, but just before waiting for the inner core worker shutdown
    fn on_shutdown(&self, _sdk_worker: &Worker) {}
    /// Called every time a workflow is about to be activated
    async fn on_workflow_activation(
        &self,
        _activation: &WorkflowActivation,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

/// Supports the composition of interceptors
pub struct InterceptorWithNext {
    inner: Box<dyn WorkerInterceptor>,
    next: Option<Box<InterceptorWithNext>>,
}

impl InterceptorWithNext {
    /// Create from an existing interceptor, can be used to initialize a chain of interceptors
    pub fn new(inner: Box<dyn WorkerInterceptor>) -> Self {
        Self { inner, next: None }
    }

    /// Sets the next interceptor, and then returns that interceptor, wrapped by
    /// [InterceptorWithNext]. You can keep calling this method on it to extend the chain.
    pub fn set_next(&mut self, next: Box<dyn WorkerInterceptor>) -> &mut InterceptorWithNext {
        self.next.insert(Box::new(Self::new(next)))
    }
}

#[async_trait::async_trait(?Send)]
impl WorkerInterceptor for InterceptorWithNext {
    async fn on_workflow_activation_completion(&self, c: &WorkflowActivationCompletion) {
        self.inner.on_workflow_activation_completion(c).await;
        if let Some(next) = &self.next {
            next.on_workflow_activation_completion(c).await;
        }
    }

    fn on_shutdown(&self, w: &Worker) {
        self.inner.on_shutdown(w);
        if let Some(next) = &self.next {
            next.on_shutdown(w);
        }
    }

    async fn on_workflow_activation(&self, a: &WorkflowActivation) -> Result<(), anyhow::Error> {
        self.inner.on_workflow_activation(a).await?;
        if let Some(next) = &self.next {
            next.on_workflow_activation(a).await?;
        }
        Ok(())
    }
}

/// An interceptor which causes the worker's run function to exit early if nondeterminism errors are
/// encountered
pub struct FailOnNondeterminismInterceptor {}
#[async_trait::async_trait(?Send)]
impl WorkerInterceptor for FailOnNondeterminismInterceptor {
    async fn on_workflow_activation(
        &self,
        activation: &WorkflowActivation,
    ) -> Result<(), anyhow::Error> {
        if matches!(
            activation.eviction_reason(),
            Some(EvictionReason::Nondeterminism)
        ) {
            bail!("Workflow is being evicted because of nondeterminism! {activation}");
        }
        Ok(())
    }
}

/// An interceptor that allows you to fetch the exit value of the workflow if and when it is set
#[derive(Default)]
pub struct ReturnWorkflowExitValueInterceptor {
    result_value: Arc<OnceLock<Payload>>,
}

impl ReturnWorkflowExitValueInterceptor {
    /// Can be used to fetch the workflow result if/when it is determined
    pub fn result_handle(&self) -> Arc<OnceLock<Payload>> {
        self.result_value.clone()
    }
}

#[async_trait::async_trait(?Send)]
impl WorkerInterceptor for ReturnWorkflowExitValueInterceptor {
    async fn on_workflow_activation_completion(&self, c: &WorkflowActivationCompletion) {
        if let Some(v) = c.complete_workflow_execution_value() {
            let _ = self.result_value.set(v.clone());
        }
    }
}

/// Boxed cancellable future used by workflow interceptor operation outputs.
pub struct BoxedCancellableFuture<T> {
    inner: Pin<Box<dyn CancellableFuture<T>>>,
}

impl<T> BoxedCancellableFuture<T> {
    pub(crate) fn new<F>(future: F) -> Self
    where
        F: CancellableFuture<T> + 'static,
    {
        Self {
            inner: Box::pin(future),
        }
    }
}

impl<T> Future for BoxedCancellableFuture<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.poll_unpin(cx)
    }
}

impl<T> futures_util::future::FusedFuture for BoxedCancellableFuture<T> {
    fn is_terminated(&self) -> bool {
        self.inner.is_terminated()
    }
}

impl<T> CancellableFuture<T> for BoxedCancellableFuture<T> {
    fn cancel(&self) {
        self.inner.cancel();
    }
}

/// SDK-owned erased typed value carried through the workflow interceptor chain.
///
/// Wraps a concrete `T: TemporalSerializable + 'static` together with `TypeId::of::<T>()` so
/// interceptors that know the concrete type can read or mutate the value.
///
/// For operations that originate as payloads (e.g. signals from history), the wrapped
/// value can be a `RawValue` so the chain still has a single uniform shape.
pub struct WorkflowValue {
    type_id: TypeId,
    inner: Box<dyn TemporalSerializable>,
}

impl WorkflowValue {
    // Intentionally private to prevent end users from `std::mem::swap`ing in differently typed
    // values in interceptors.
    pub(crate) fn new<T>(value: T) -> Self
    where
        T: TemporalSerializable + 'static,
    {
        Self {
            type_id: TypeId::of::<T>(),
            inner: Box::new(value),
        }
    }

    /// `TypeId` of the wrapped concrete type.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Returns true if the wrapped value's concrete type is `T`.
    pub fn is<T>(&self) -> bool
    where
        T: TemporalSerializable + 'static,
    {
        self.type_id == TypeId::of::<T>()
    }

    /// Borrow the wrapped value as `&T` if its concrete type matches.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: TemporalSerializable + 'static,
    {
        if self.is::<T>() {
            // SAFETY: `WorkflowValue` invariant: `type_id` is `TypeId::of::<T>()` for the
            // concrete type stored in `inner`. Casting `*const dyn TemporalSerializable` to
            // `*const T` discards the vtable and yields the data pointer to that concrete
            // value, which is what we return a reference to.
            let ptr = self.inner.as_ref() as *const dyn TemporalSerializable as *const T;
            Some(unsafe { &*ptr })
        } else {
            None
        }
    }

    /// Mutably borrow the wrapped value as `&mut T` if its concrete type matches.
    ///
    /// This is the only way for an interceptor to change the carried value — assignment
    /// through this reference replaces the value in place while preserving the type the SDK
    /// expects to recover later.
    pub fn downcast_mut<T>(&mut self) -> Option<&mut T>
    where
        T: TemporalSerializable + 'static,
    {
        if self.is::<T>() {
            // SAFETY: see `downcast_ref`. Same invariant; `&mut` is sound because we hold a
            // unique borrow of `self`.
            let ptr = self.inner.as_mut() as *mut dyn TemporalSerializable as *mut T;
            Some(unsafe { &mut *ptr })
        } else {
            None
        }
    }

    pub(crate) fn into_typed<T>(self) -> Result<T, Self>
    where
        T: TemporalSerializable + 'static,
    {
        if self.is::<T>() {
            // SAFETY: invariant guarantees `inner` was constructed by us as `Box::new(value)`
            // for some `T`-typed value, so the layout matches.
            let raw: *mut dyn TemporalSerializable = Box::into_raw(self.inner);
            Ok(*unsafe { Box::from_raw(raw as *mut T) })
        } else {
            Err(self)
        }
    }
}

impl fmt::Debug for WorkflowValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkflowValue")
            .field("type_id", &self.type_id)
            .finish_non_exhaustive()
    }
}

/// Output type for workflow execution interception.
pub type WorkflowExecuteOutput = LocalBoxFuture<'static, WorkflowResult<Payload>>;

/// Output type for workflow signal interception.
pub type WorkflowSignalOutput = LocalBoxFuture<'static, Result<(), WorkflowError>>;

/// Output type for workflow query interception.
pub type WorkflowQueryOutput = Result<Payload, WorkflowError>;

/// Output type for workflow timer interception.
pub type SleepOutput = BoxedCancellableFuture<TimerResult>;

/// Continuation for an interceptor operation.
///
/// Async behavior is still supported by composing on the *returned* operation output. For
/// operations whose output is a future, the interceptor can `Box::pin(async move { ... })` over
/// the future returned by `next.run(input)` to observe completion or transform the result.
#[must_use = "workflow interceptor continuations must be run to continue the call chain"]
pub struct Next<'a, I, O> {
    inner: Box<dyn FnOnce(I) -> O + 'a>,
}

impl<'a, I, O> Next<'a, I, O> {
    pub(crate) fn new(f: impl FnOnce(I) -> O + 'a) -> Self {
        Self { inner: Box::new(f) }
    }

    /// Continue the call chain with the provided input.
    ///
    /// Must be called synchronously inside the interceptor method body. For async-output
    /// operations, the interceptor should call `run` first to obtain the operation future, then
    /// return a composed future that wraps it.
    #[must_use = "the returned workflow interceptor output must be used"]
    pub fn run(self, input: I) -> O {
        (self.inner)(input)
    }
}

/// Read-only context passed to workflow interceptors.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WorkflowInterceptorContext {
    /// Read-only workflow metadata.
    pub workflow: WorkflowContextView,
    /// Operation-specific metadata.
    pub operation: WorkflowOperationContext,
}

/// Operation-specific workflow interceptor context.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WorkflowOperationContext {
    /// Raw replay state for the current activation/task as observed from Core.
    pub is_replaying: bool,
    /// True when this operation is being executed only to replay history events.
    pub is_replaying_history_events: bool,
}

impl WorkflowOperationContext {
    pub(crate) fn new(is_replaying: bool, is_replaying_history_events: bool) -> Self {
        Self {
            is_replaying,
            is_replaying_history_events,
        }
    }
}

/// Input for workflow execution interception.
#[derive(Debug)]
#[non_exhaustive]
pub struct ExecuteInput {
    workflow_type: String,
    args: WorkflowValue,
    headers: HashMap<String, Payload>,
    context: WorkflowInterceptorContext,
}

impl ExecuteInput {
    pub(crate) fn new(
        workflow_type: String,
        args: WorkflowValue,
        headers: HashMap<String, Payload>,
        context: WorkflowInterceptorContext,
    ) -> Self {
        Self {
            workflow_type,
            args,
            headers,
            context,
        }
    }

    /// Workflow type being executed.
    pub fn workflow_type(&self) -> &str {
        &self.workflow_type
    }

    /// Workflow arguments as an erased typed value.
    ///
    /// The carried concrete type is always the workflow's typed `Input` regardless of
    /// whether the workflow uses a separate `#[init]` method. Use
    /// [`WorkflowValue::downcast_ref`] / [`WorkflowValue::downcast_mut`] when the
    /// interceptor knows that type. Mutating the args in place changes what the workflow
    /// handler observes — for non-split workflows that's `W::run`, for split-init
    /// workflows it's `W::init`. The SDK runs the chain at workflow construction time and
    /// reads the (possibly-mutated) args back out before invoking whichever handler
    /// consumes them.
    pub fn args(&self) -> &WorkflowValue {
        &self.args
    }

    /// Mutable access to the workflow arguments.
    pub fn args_mut(&mut self) -> &mut WorkflowValue {
        &mut self.args
    }

    /// Workflow headers.
    pub fn headers(&self) -> &HashMap<String, Payload> {
        &self.headers
    }

    /// Read-only workflow interceptor context.
    pub fn context(&self) -> &WorkflowInterceptorContext {
        &self.context
    }

    pub(crate) fn into_args(self) -> WorkflowValue {
        self.args
    }
}

/// Input for workflow signal interception.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HandleSignalInput {
    signal_name: String,
    args: Vec<Payload>,
    headers: HashMap<String, Payload>,
    context: WorkflowInterceptorContext,
}

impl HandleSignalInput {
    pub(crate) fn new(
        signal_name: String,
        args: Vec<Payload>,
        headers: HashMap<String, Payload>,
        context: WorkflowInterceptorContext,
    ) -> Self {
        Self {
            signal_name,
            args,
            headers,
            context,
        }
    }

    /// Signal name being handled.
    pub fn signal_name(&self) -> &str {
        &self.signal_name
    }

    /// Serialized signal arguments.
    pub fn args(&self) -> &[Payload] {
        &self.args
    }

    /// Signal headers.
    pub fn headers(&self) -> &HashMap<String, Payload> {
        &self.headers
    }

    /// Read-only workflow interceptor context.
    pub fn context(&self) -> &WorkflowInterceptorContext {
        &self.context
    }

    pub(crate) fn into_parts(self) -> (String, Payloads, HashMap<String, Payload>) {
        (
            self.signal_name,
            Payloads {
                payloads: self.args,
            },
            self.headers,
        )
    }
}

/// Input for workflow query interception.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HandleQueryInput {
    query_name: String,
    args: Vec<Payload>,
    headers: HashMap<String, Payload>,
    context: WorkflowInterceptorContext,
}

impl HandleQueryInput {
    pub(crate) fn new(
        query_name: String,
        args: Vec<Payload>,
        headers: HashMap<String, Payload>,
        context: WorkflowInterceptorContext,
    ) -> Self {
        Self {
            query_name,
            args,
            headers,
            context,
        }
    }

    /// Query name being handled.
    pub fn query_name(&self) -> &str {
        &self.query_name
    }

    /// Serialized query arguments.
    pub fn args(&self) -> &[Payload] {
        &self.args
    }

    /// Query headers.
    pub fn headers(&self) -> &HashMap<String, Payload> {
        &self.headers
    }

    /// Read-only workflow interceptor context.
    pub fn context(&self) -> &WorkflowInterceptorContext {
        &self.context
    }

    pub(crate) fn into_parts(self) -> (String, Payloads, HashMap<String, Payload>) {
        (
            self.query_name,
            Payloads {
                payloads: self.args,
            },
            self.headers,
        )
    }
}

/// Input for workflow timer interception.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SleepInput {
    options: TimerOptions,
    context: WorkflowInterceptorContext,
}

impl SleepInput {
    pub(crate) fn new(options: TimerOptions, context: WorkflowInterceptorContext) -> Self {
        Self { options, context }
    }

    /// Timer duration.
    pub fn duration(&self) -> std::time::Duration {
        self.options.duration
    }

    /// Timer summary.
    pub fn summary(&self) -> Option<&str> {
        self.options.summary.as_deref()
    }

    /// Read-only workflow interceptor context.
    pub fn context(&self) -> &WorkflowInterceptorContext {
        &self.context
    }

    /// Return a copy of this input with a different timer duration.
    pub fn with_duration(mut self, duration: std::time::Duration) -> Self {
        self.options.duration = duration;
        self
    }

    /// Return a copy of this input with a different timer summary.
    pub fn with_summary(mut self, summary: impl Into<Option<String>>) -> Self {
        self.options.summary = summary.into();
        self
    }

    pub(crate) fn into_options(self) -> TimerOptions {
        self.options
    }
}

/// Factory trait for workflow interceptors.
pub trait WorkflowInterceptor: Send + Sync + 'static {
    /// Build interceptors for a workflow instance.
    fn intercept_workflow(&self, ctx: WorkflowInterceptorContext) -> WorkflowInterceptors;
}

/// Inbound and outbound interceptors for one workflow instance.
pub struct WorkflowInterceptors {
    /// Inbound workflow interceptor.
    pub inbound: Box<dyn WorkflowInboundInterceptor>,
    /// Outbound workflow interceptor.
    pub outbound: Box<dyn WorkflowOutboundInterceptor>,
}

impl Default for WorkflowInterceptors {
    fn default() -> Self {
        Self {
            inbound: Box::new(NoopWorkflowInboundInterceptor),
            outbound: Box::new(NoopWorkflowOutboundInterceptor),
        }
    }
}

/// Inbound workflow interceptor hooks.
///
/// Methods are synchronous (`fn`, not `async fn`). For operations whose output is a future
/// (e.g. [`WorkflowExecuteOutput`], [`WorkflowSignalOutput`]), the implementor must call
/// `next.run(input)` synchronously to obtain the operation future, then return either that
/// future or a composed future that wraps it. Awaiting before calling `next` would insert a
/// yield point into the workflow coroutine and break replay determinism, and is structurally
/// prevented by the [`Next`] type.
pub trait WorkflowInboundInterceptor: Send + Sync + 'static {
    /// Intercept workflow execution.
    fn execute<'a>(
        &'a self,
        input: ExecuteInput,
        next: Next<'a, ExecuteInput, WorkflowExecuteOutput>,
    ) -> WorkflowExecuteOutput {
        next.run(input)
    }

    /// Intercept workflow signal handling.
    fn handle_signal<'a>(
        &'a self,
        input: HandleSignalInput,
        next: Next<'a, HandleSignalInput, WorkflowSignalOutput>,
    ) -> WorkflowSignalOutput {
        next.run(input)
    }

    /// Intercept workflow query handling.
    fn handle_query<'a>(
        &'a self,
        input: HandleQueryInput,
        next: Next<'a, HandleQueryInput, WorkflowQueryOutput>,
    ) -> WorkflowQueryOutput {
        next.run(input)
    }
}

/// Outbound workflow interceptor hooks.
///
/// Methods are synchronous (`fn`, not `async fn`). The output type for each hook matches the
/// SDK operation it wraps: synchronous operations return values directly, asynchronous
/// operations return futures (typically [`BoxedCancellableFuture`] or `LocalBoxFuture`).
/// Implementors must call `next.run(input)` synchronously before returning; async observation
/// of operation completion is done by wrapping the returned future. The same replay-determinism
/// rationale as [`WorkflowInboundInterceptor`] applies.
pub trait WorkflowOutboundInterceptor: Send + Sync + 'static {
    /// Intercept workflow timer creation.
    fn sleep<'a>(
        &'a self,
        input: SleepInput,
        next: Next<'a, SleepInput, SleepOutput>,
    ) -> SleepOutput {
        next.run(input)
    }
}

struct NoopWorkflowInboundInterceptor;

impl WorkflowInboundInterceptor for NoopWorkflowInboundInterceptor {}

struct NoopWorkflowOutboundInterceptor;

impl WorkflowOutboundInterceptor for NoopWorkflowOutboundInterceptor {}

#[derive(Clone, Default)]
pub(crate) struct WorkflowInterceptorInstance {
    inbound: Rc<Vec<Rc<dyn WorkflowInboundInterceptor>>>,
    outbound: Rc<Vec<Rc<dyn WorkflowOutboundInterceptor>>>,
}

impl WorkflowInterceptorInstance {
    pub(crate) fn new(
        interceptors: &[Arc<dyn WorkflowInterceptor>],
        ctx: WorkflowInterceptorContext,
    ) -> Self {
        let mut inbound = Vec::with_capacity(interceptors.len());
        let mut outbound = Vec::with_capacity(interceptors.len());
        for interceptor in interceptors {
            let WorkflowInterceptors {
                inbound: next_inbound,
                outbound: next_outbound,
            } = interceptor.intercept_workflow(ctx.clone());
            inbound.push(Rc::from(next_inbound));
            outbound.push(Rc::from(next_outbound));
        }
        Self {
            inbound: Rc::new(inbound),
            outbound: Rc::new(outbound),
        }
    }

    pub(crate) fn execute(
        &self,
        input: ExecuteInput,
        next: impl FnOnce(ExecuteInput) -> WorkflowExecuteOutput,
    ) -> WorkflowExecuteOutput {
        call_execute(&self.inbound, input, Next::new(next))
    }

    pub(crate) fn handle_signal(
        &self,
        input: HandleSignalInput,
        next: impl FnOnce(HandleSignalInput) -> WorkflowSignalOutput,
    ) -> WorkflowSignalOutput {
        call_handle_signal(&self.inbound, input, Next::new(next))
    }

    pub(crate) fn handle_query(
        &self,
        input: HandleQueryInput,
        next: impl FnOnce(HandleQueryInput) -> WorkflowQueryOutput,
    ) -> WorkflowQueryOutput {
        call_handle_query(&self.inbound, input, Next::new(next))
    }

    pub(crate) fn sleep(
        &self,
        input: SleepInput,
        next: impl FnOnce(SleepInput) -> SleepOutput,
    ) -> SleepOutput {
        call_sleep(&self.outbound, input, Next::new(next))
    }
}

macro_rules! workflow_interceptor_call {
    ($call_fn:ident, $interceptor_trait:ident, $method:ident, $input:ty, $output:ty) => {
        fn $call_fn<'a>(
            interceptors: &'a [Rc<dyn $interceptor_trait>],
            input: $input,
            next: Next<'a, $input, $output>,
        ) -> $output {
            if let Some((first, rest)) = interceptors.split_first() {
                first.$method(input, Next::new(move |input| $call_fn(rest, input, next)))
            } else {
                next.run(input)
            }
        }
    };
}

workflow_interceptor_call!(
    call_execute,
    WorkflowInboundInterceptor,
    execute,
    ExecuteInput,
    WorkflowExecuteOutput
);
workflow_interceptor_call!(
    call_handle_signal,
    WorkflowInboundInterceptor,
    handle_signal,
    HandleSignalInput,
    WorkflowSignalOutput
);
workflow_interceptor_call!(
    call_handle_query,
    WorkflowInboundInterceptor,
    handle_query,
    HandleQueryInput,
    WorkflowQueryOutput
);
workflow_interceptor_call!(
    call_sleep,
    WorkflowOutboundInterceptor,
    sleep,
    SleepInput,
    SleepOutput
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::Mutex,
        task::{Context, Poll},
        time::Duration,
    };

    fn test_context() -> WorkflowInterceptorContext {
        WorkflowInterceptorContext {
            workflow: WorkflowContextView {
                workflow_id: "wf-id".to_string(),
                run_id: "run-id".to_string(),
                workflow_type: "TestWorkflow".to_string(),
                task_queue: "task-queue".to_string(),
                namespace: "default".to_string(),
                attempt: 1,
                first_execution_run_id: "run-id".to_string(),
                continued_from_run_id: None,
                start_time: None,
                execution_timeout: None,
                run_timeout: None,
                task_timeout: None,
                parent: None,
                root: None,
                retry_policy: None,
                cron_schedule: None,
                memo: None,
                search_attributes: None,
            },
            operation: WorkflowOperationContext::new(false, false),
        }
    }

    struct RecordingWorkflowInterceptor {
        name: &'static str,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl WorkflowInterceptor for RecordingWorkflowInterceptor {
        fn intercept_workflow(&self, _ctx: WorkflowInterceptorContext) -> WorkflowInterceptors {
            WorkflowInterceptors {
                inbound: Box::new(RecordingInboundInterceptor {
                    name: self.name,
                    events: self.events.clone(),
                }),
                outbound: Box::new(NoopWorkflowOutboundInterceptor),
            }
        }
    }

    struct RecordingInboundInterceptor {
        name: &'static str,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl WorkflowInboundInterceptor for RecordingInboundInterceptor {
        fn handle_query<'a>(
            &'a self,
            input: HandleQueryInput,
            next: Next<'a, HandleQueryInput, WorkflowQueryOutput>,
        ) -> WorkflowQueryOutput {
            self.events
                .lock()
                .unwrap()
                .push(format!("{} before", self.name));
            let result = next.run(input);
            self.events
                .lock()
                .unwrap()
                .push(format!("{} after", self.name));
            result
        }
    }

    #[test]
    fn inbound_interceptors_run_in_registration_order() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let interceptors: Vec<Arc<dyn WorkflowInterceptor>> = vec![
            Arc::new(RecordingWorkflowInterceptor {
                name: "first",
                events: events.clone(),
            }),
            Arc::new(RecordingWorkflowInterceptor {
                name: "second",
                events: events.clone(),
            }),
        ];
        let instance = WorkflowInterceptorInstance::new(&interceptors, test_context());
        let input =
            HandleQueryInput::new("query".to_string(), vec![], HashMap::new(), test_context());

        let result = instance.handle_query(input, |_| {
            events.lock().unwrap().push("next".to_string());
            Ok(Payload::default())
        });

        assert!(result.is_ok());
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "first before",
                "second before",
                "next",
                "second after",
                "first after"
            ]
        );
    }

    struct SleepMutatingWorkflowInterceptor {
        duration: Duration,
    }

    impl WorkflowInterceptor for SleepMutatingWorkflowInterceptor {
        fn intercept_workflow(&self, _ctx: WorkflowInterceptorContext) -> WorkflowInterceptors {
            WorkflowInterceptors {
                inbound: Box::new(NoopWorkflowInboundInterceptor),
                outbound: Box::new(SleepMutatingOutboundInterceptor {
                    duration: self.duration,
                }),
            }
        }
    }

    struct SleepMutatingOutboundInterceptor {
        duration: Duration,
    }

    impl WorkflowOutboundInterceptor for SleepMutatingOutboundInterceptor {
        fn sleep<'a>(
            &'a self,
            input: SleepInput,
            next: Next<'a, SleepInput, SleepOutput>,
        ) -> SleepOutput {
            next.run(input.with_duration(self.duration))
        }
    }

    struct ReadyCancellableFuture;

    impl Future for ReadyCancellableFuture {
        type Output = TimerResult;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(TimerResult::Fired)
        }
    }

    impl futures_util::future::FusedFuture for ReadyCancellableFuture {
        fn is_terminated(&self) -> bool {
            false
        }
    }

    impl CancellableFuture<TimerResult> for ReadyCancellableFuture {
        fn cancel(&self) {}
    }

    #[test]
    fn outbound_sleep_interceptor_can_forward_modified_input() {
        let interceptors: Vec<Arc<dyn WorkflowInterceptor>> =
            vec![Arc::new(SleepMutatingWorkflowInterceptor {
                duration: Duration::from_secs(3),
            })];
        let instance = WorkflowInterceptorInstance::new(&interceptors, test_context());
        let observed_duration = Arc::new(Mutex::new(None));
        let observed_duration_for_next = observed_duration.clone();
        let input = SleepInput::new(
            TimerOptions {
                duration: Duration::from_secs(1),
                summary: None,
            },
            test_context(),
        );

        let _output = instance.sleep(input, move |input| {
            observed_duration_for_next
                .lock()
                .unwrap()
                .replace(input.duration());
            BoxedCancellableFuture::new(ReadyCancellableFuture)
        });

        assert_eq!(
            *observed_duration.lock().unwrap(),
            Some(Duration::from_secs(3))
        );
    }

    struct SignalWrappingWorkflowInterceptor {
        observed: Arc<Mutex<Option<&'static str>>>,
    }

    impl WorkflowInterceptor for SignalWrappingWorkflowInterceptor {
        fn intercept_workflow(&self, _ctx: WorkflowInterceptorContext) -> WorkflowInterceptors {
            WorkflowInterceptors {
                inbound: Box::new(SignalWrappingInbound {
                    observed: self.observed.clone(),
                }),
                outbound: Box::new(NoopWorkflowOutboundInterceptor),
            }
        }
    }

    struct SignalWrappingInbound {
        observed: Arc<Mutex<Option<&'static str>>>,
    }

    impl WorkflowInboundInterceptor for SignalWrappingInbound {
        fn handle_signal<'a>(
            &'a self,
            input: HandleSignalInput,
            next: Next<'a, HandleSignalInput, WorkflowSignalOutput>,
        ) -> WorkflowSignalOutput {
            let inner = next.run(input);
            let observed = self.observed.clone();
            Box::pin(async move {
                let result = inner.await;
                observed
                    .lock()
                    .unwrap()
                    .replace(if result.is_ok() { "ok" } else { "err" });
                result
            })
        }
    }

    #[test]
    fn inbound_signal_interceptor_can_wrap_output_future() {
        let observed = Arc::new(Mutex::new(None));
        let interceptors: Vec<Arc<dyn WorkflowInterceptor>> =
            vec![Arc::new(SignalWrappingWorkflowInterceptor {
                observed: observed.clone(),
            })];
        let instance = WorkflowInterceptorInstance::new(&interceptors, test_context());
        let input =
            HandleSignalInput::new("sig".to_string(), vec![], HashMap::new(), test_context());

        let fut = instance.handle_signal(input, |_| Box::pin(async { Ok(()) }));

        assert!(observed.lock().unwrap().is_none());
        let result = futures::executor::block_on(fut);
        assert!(result.is_ok());
        assert_eq!(*observed.lock().unwrap(), Some("ok"));
    }

    #[test]
    fn workflow_value_downcast_ref_and_mut() {
        let mut value = WorkflowValue::new(42_i32);
        assert!(value.is::<i32>());
        assert!(!value.is::<String>());
        assert_eq!(value.downcast_ref::<i32>(), Some(&42));
        assert!(value.downcast_ref::<String>().is_none());

        *value.downcast_mut::<i32>().unwrap() = 7;
        assert_eq!(value.downcast_ref::<i32>(), Some(&7));
    }

    struct ArgsMutatingExecuteInterceptor {
        replacement: i32,
    }

    impl WorkflowInterceptor for ArgsMutatingExecuteInterceptor {
        fn intercept_workflow(&self, _ctx: WorkflowInterceptorContext) -> WorkflowInterceptors {
            WorkflowInterceptors {
                inbound: Box::new(ArgsMutatingExecuteInbound {
                    replacement: self.replacement,
                }),
                outbound: Box::new(NoopWorkflowOutboundInterceptor),
            }
        }
    }

    struct ArgsMutatingExecuteInbound {
        replacement: i32,
    }

    impl WorkflowInboundInterceptor for ArgsMutatingExecuteInbound {
        fn execute<'a>(
            &'a self,
            mut input: ExecuteInput,
            next: Next<'a, ExecuteInput, WorkflowExecuteOutput>,
        ) -> WorkflowExecuteOutput {
            *input
                .args_mut()
                .downcast_mut::<i32>()
                .expect("execute input should carry the workflow's typed input") = self.replacement;
            next.run(input)
        }
    }

    #[test]
    fn execute_interceptor_arg_mutation_flows_to_handler() {
        let observed = Arc::new(Mutex::new(None));
        let observed_for_next = observed.clone();
        let interceptors: Vec<Arc<dyn WorkflowInterceptor>> =
            vec![Arc::new(ArgsMutatingExecuteInterceptor { replacement: 99 })];
        let instance = WorkflowInterceptorInstance::new(&interceptors, test_context());

        let input = ExecuteInput::new(
            "TestWorkflow".to_string(),
            WorkflowValue::new(1_i32),
            HashMap::new(),
            test_context(),
        );

        let _ = instance.execute(input, move |input| {
            let typed = input.into_args().into_typed::<i32>().unwrap();
            observed_for_next.lock().unwrap().replace(typed);
            Box::pin(async { Ok(Payload::default()) })
        });

        assert_eq!(*observed.lock().unwrap(), Some(99));
    }
}
