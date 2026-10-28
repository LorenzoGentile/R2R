"""Opt-in, content-free receipts for attributed LiteLLM ingestion calls."""

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Iterator, Literal, Optional
from uuid import UUID, uuid4

import litellm
from litellm.integrations.custom_logger import CustomLogger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

_logger = logging.getLogger(__name__)
_TokenCount = Annotated[int, Field(strict=True, ge=0)]
_DIRECTORY_MODE = 0o700
_OPENAI_MODEL_PREFIX = "openai/"
_POST_API_CALL = "post_api_call"


class UsageOperation(str, Enum):
    """Identify the ingestion work responsible for a model call."""

    DOCUMENT_SUMMARY = "document_summary"
    SUMMARY_EMBEDDING = "summary_embedding"
    CHUNK_EMBEDDING = "chunk_embedding"
    CHUNK_ENRICHMENT = "chunk_enrichment"
    ENRICHMENT_EMBEDDING = "enrichment_embedding"


class UsageStatus(str, Enum):
    """Distinguish observed counters from accounting gaps."""

    OBSERVED = "observed"
    MISSING = "missing"
    INVALID = "invalid"


class UsageSource(str, Enum):
    """Identify whether counters precede SDK normalization."""

    PROVIDER_RESPONSE = "provider_response"
    LITELLM = "litellm"


class UsageContext(BaseModel):
    """Correlate a service-stage execution with R2R document identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    owner_id: UUID
    collection_ids: tuple[UUID, ...] = ()
    operation: UsageOperation


class UsageTokens(BaseModel):
    """Store inclusive input tokens with cache counts as subsets, not additions."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: Optional[_TokenCount] = None
    output_tokens: Optional[_TokenCount] = None
    cache_read_input_tokens: Optional[_TokenCount] = None
    cache_write_input_tokens: Optional[_TokenCount] = None

    @model_validator(mode="after")
    def validate_cache_counts(self) -> "UsageTokens":
        """Reject cache counts that exceed the inclusive input count."""
        cached = (self.cache_read_input_tokens or 0) + (
            self.cache_write_input_tokens or 0
        )
        if self.input_tokens is not None and cached > self.input_tokens:
            raise ValueError("Cache tokens exceed input tokens")
        return self


class UsageReceipt(BaseModel):
    """Provide an immutable event identity for replay-safe external accounting."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    context: UsageContext
    adapter: Literal["litellm"] = "litellm"
    provider: Optional[str] = None
    requested_model: Optional[str] = None
    response_model: Optional[str] = None
    response_id: Optional[str] = None
    call_failed: bool = False
    usage_status: UsageStatus = UsageStatus.MISSING
    usage_source: UsageSource = UsageSource.LITELLM
    tokens: UsageTokens = Field(default_factory=UsageTokens)


class UsageCall(BaseModel):
    """Hold the external SDK response until the call boundary is recorded."""

    response: Any = None
    recording: bool = False
    capture_raw_usage: bool = False
    raw_tokens: Optional[UsageTokens] = None
    raw_usage_status: Optional[UsageStatus] = None

    def observe(self, details: Any) -> None:
        """Read raw OpenAI-compatible counters before LiteLLM fills missing zeros."""
        event = _LogDetails.model_validate(details)
        if self.capture_raw_usage and event.log_event_type == _POST_API_CALL:
            try:
                response = (
                    _Response.model_validate_json(event.original_response)
                    if isinstance(event.original_response, str)
                    else _Response.model_validate(event.original_response)
                )
                self.raw_tokens = _usage_tokens(response.usage)
                self.raw_usage_status = _usage_status(self.raw_tokens)
            except ValidationError:
                self.raw_usage_status = UsageStatus.INVALID


_context: ContextVar[Optional[UsageContext]] = ContextVar(
    "r2r_usage_context", default=None
)
_active_call: ContextVar[Optional[UsageCall]] = ContextVar(
    "r2r_usage_call", default=None
)


@contextmanager
def usage_context(context: UsageContext) -> Iterator[None]:
    """Bind attribution to the current task without leaking it to other jobs."""
    token = _context.set(context)
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def capture_usage(
    directory: Optional[str], requested_model: Optional[str]
) -> Iterator[UsageCall]:
    """Persist one receipt per attributed provider attempt without retrying writes."""
    call = UsageCall()
    context = _context.get()
    if directory is None or context is None:
        yield call
        return

    receipt = UsageReceipt(context=context, requested_model=requested_model)
    call.recording = True
    # shortcut: other routes use normalized counters, qualify raw usage before billing them.
    call.capture_raw_usage = bool(
        requested_model and requested_model.startswith(_OPENAI_MODEL_PREFIX)
    )
    if _usage_logger not in litellm.input_callback:
        litellm.input_callback.append(_usage_logger)
    token = _active_call.set(call)
    try:
        yield call
    except BaseException as exc:
        receipt.call_failed = True
        if call.response is None:
            call.response = getattr(exc, "response", None)
        raise
    finally:
        _active_call.reset(token)
        try:
            _populate_receipt(receipt, call.response)
            if call.raw_usage_status is not None:
                receipt.usage_status = call.raw_usage_status
                receipt.usage_source = UsageSource.PROVIDER_RESPONSE
                receipt.tokens = call.raw_tokens or UsageTokens()
            _publish_receipt(Path(directory), receipt)
        except Exception:
            # Accounting failure must not retry an already paid model call.
            _logger.critical(
                "Usage receipt persistence failed; accounting gap for event %s",
                receipt.event_id,
            )


class _PromptDetails(BaseModel):
    """Read the normalized LiteLLM prompt-cache counters."""

    model_config = ConfigDict(from_attributes=True)
    cached_tokens: Optional[_TokenCount] = None


class _ResponseUsage(BaseModel):
    """Validate the known part of a LiteLLM usage response."""

    model_config = ConfigDict(from_attributes=True)
    prompt_tokens: Optional[_TokenCount] = None
    completion_tokens: Optional[_TokenCount] = None
    prompt_tokens_details: Optional[_PromptDetails] = None
    cache_creation_input_tokens: Optional[_TokenCount] = None


class _ResponseIdentity(BaseModel):
    """Read response identity independently of possibly invalid usage."""

    model_config = ConfigDict(from_attributes=True)
    model: Optional[str] = None
    id: Optional[str] = None


class _Response(_ResponseIdentity):
    """Read model identity and usage without copying response content."""

    usage: Optional[_ResponseUsage] = None


class _LogDetails(BaseModel):
    """Select the post-call response from LiteLLM's unstructured logging payload."""

    log_event_type: Optional[str] = None
    original_response: Any = None


class _ResponseMetadata(BaseModel):
    """Read only provider identity from private LiteLLM routing metadata."""

    custom_llm_provider: Optional[str] = None


class _UsageLogger(CustomLogger):
    """Read raw counters synchronously using only task-local call state."""

    def log_post_api_call(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Observe the response before LiteLLM's response normalization."""
        call = _active_call.get()
        if call is not None:
            call.observe(kwargs)


# The callback is a singleton; request data lives only in ContextVars.
_usage_logger = _UsageLogger()


def _populate_receipt(receipt: UsageReceipt, response: Any) -> None:
    """Extract validated normalized counters, preserving missing or invalid usage."""
    if response is None:
        return
    try:
        identity = _ResponseIdentity.model_validate(response)
        receipt.response_model = identity.model
        receipt.response_id = identity.id
        metadata = _ResponseMetadata.model_validate(
            getattr(response, "_hidden_params", {})
        )
        receipt.provider = metadata.custom_llm_provider
        parsed = _Response.model_validate(response)
        if parsed.usage is None:
            return
        usage = parsed.usage
        cache_write = usage.cache_creation_input_tokens
        if cache_write is None:
            # LiteLLM 1.x keeps Anthropic cache writes in a private attribute.
            cache_write = getattr(
                getattr(response, "usage", None),
                "_cache_creation_input_tokens",
                None,
            )
        receipt.tokens = _usage_tokens(usage, cache_write)
        receipt.usage_status = _usage_status(receipt.tokens)
    except ValidationError:
        receipt.usage_status = UsageStatus.INVALID
        _logger.error("Invalid model usage for event %s", receipt.event_id)


def _usage_tokens(
    usage: Optional[_ResponseUsage], cache_write: Optional[int] = None
) -> UsageTokens:
    """Keep absent counters unknown and validate inclusive cache semantics."""
    if usage is None:
        return UsageTokens()
    return UsageTokens(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_input_tokens=(
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details is not None
            else None
        ),
        cache_write_input_tokens=(
            usage.cache_creation_input_tokens
            if usage.cache_creation_input_tokens is not None
            else cache_write
        ),
    )


def _usage_status(tokens: UsageTokens) -> UsageStatus:
    """Distinguish an observed zero from the absence of all counters."""
    if any(value is not None for value in tokens.model_dump().values()):
        return UsageStatus.OBSERVED
    return UsageStatus.MISSING


def _publish_receipt(directory: Path, receipt: UsageReceipt) -> None:
    """Flush a private file before atomically publishing it for an importer."""
    directory.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(receipt.model_dump_json())
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(directory / f"{receipt.event_id}.json")
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
