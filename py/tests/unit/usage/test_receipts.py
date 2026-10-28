"""Exercise ingestion accounting through real providers with simulated model I/O."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
import toml
from litellm import EmbeddingResponse, ModelResponse
from litellm.types.utils import Usage
from openai import AsyncOpenAI

from core.base import (
    AppConfig,
    CompletionConfig,
    DocumentChunk,
    EmbeddingConfig,
    R2RException,
)
from core.base.abstractions import ChunkEnrichmentSettings, GenerationConfig
from core.base.usage import (
    UsageContext,
    UsageOperation,
    UsageReceipt,
    UsageSource,
    UsageStatus,
    capture_usage,
    usage_context,
)
from core.main.config import R2RConfig
from core.main.orchestration.simple.ingestion_workflow import (
    simple_ingestion_factory,
)
from core.main.services.ingestion_service import IngestionService
from core.providers.embeddings.litellm import LiteLLMEmbeddingProvider
from core.providers.llm.litellm import LiteLLMCompletionProvider
from shared.abstractions.document import DocumentResponse, DocumentType


@pytest.fixture
def context():
    """Provide attribution unrelated to user-supplied metadata."""
    return UsageContext(
        document_id=uuid4(),
        owner_id=uuid4(),
        collection_ids=(uuid4(),),
        operation=UsageOperation.CHUNK_EMBEDDING,
    )


@pytest.fixture
def providers(tmp_path):
    """Use real R2R providers and deterministic SDK responses."""
    app = AppConfig(usage_receipts_directory=str(tmp_path))
    embedding = LiteLLMEmbeddingProvider(
        EmbeddingConfig(
            app=app,
            provider="litellm",
            base_model="openai/text-embedding-3-small",
            base_dimension=2,
            initial_backoff=0,
            max_retries=2,
        )
    )
    llm = LiteLLMCompletionProvider(
        CompletionConfig(
            app=app,
            provider="litellm",
            initial_backoff=0,
            max_retries=2,
        )
    )

    async def embed(input, **kwargs):
        """Return one vector per input with known batch usage."""
        await asyncio.sleep(0)
        return EmbeddingResponse(
            model="text-embedding-3-small",
            data=[
                {"embedding": [0.25, 0.75], "index": i}
                for i in range(len(input))
            ],
            usage=Usage(
                prompt_tokens=10 * len(input), total_tokens=10 * len(input)
            ),
        )

    embedding.litellm_aembedding = AsyncMock(side_effect=embed)
    llm.acompletion = AsyncMock(
        return_value=ModelResponse(
            model="gpt-4o-mini-2024-07-18",
            id="response-fixture",
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "private summary",
                    }
                }
            ],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details={"cached_tokens": 30},
            ),
        )
    )
    yield SimpleNamespace(embedding=embedding, llm=llm)
    llm.thread_pool.shutdown()


@pytest.fixture
def service(providers):
    """Replace database access while retaining real ingestion and model adapters."""
    providers.database = SimpleNamespace(
        prompts_handler=SimpleNamespace(
            get_message_payload=AsyncMock(
                return_value=[{"role": "user", "content": "private prompt"}]
            ),
        )
    )
    return IngestionService(
        config=SimpleNamespace(
            app=SimpleNamespace(fast_llm="openai/gpt-4o-mini"),
            ingestion=SimpleNamespace(
                skip_document_summary=False,
                chunks_for_document_summary=10,
                document_summary_max_length=10000,
                document_summary_model=None,
                document_summary_system_prompt="summary",
                document_summary_task_prompt="summary",
                chunk_size=1024,
            ),
        ),
        providers=providers,
    )


def receipts(directory):
    """Read only atomically published receipts as an external importer would."""
    return [
        UsageReceipt.model_validate_json(p.read_text())
        for p in directory.glob("*.json")
    ]


def chunk(context):
    """Build a chunk whose free-form metadata attempts to override attribution."""
    return DocumentChunk(
        id=uuid4(),
        document_id=context.document_id,
        owner_id=context.owner_id,
        collection_ids=list(context.collection_ids),
        data="private document body",
        metadata={
            "document_id": str(uuid4()),
            "org_id": "forged-workspace",
            "api_key": "secret-key",
        },
    ).model_dump()


def document(context):
    """Build the document record used by summary generation."""
    return DocumentResponse(
        id=context.document_id,
        owner_id=context.owner_id,
        collection_ids=list(context.collection_ids),
        document_type=DocumentType.TXT,
        metadata={},
        version="v0",
        size_in_bytes=20,
    )


async def test_summary_and_embeddings_are_durable(service, context, tmp_path):
    """Persist replayable, content-free receipts for every ingestion operation."""
    info = document(context)
    await service.augment_document_info(info, [chunk(context)])
    entries = [
        entry
        async for entry in service.embed_document(
            [chunk(context), chunk(context)]
        )
    ]
    assert len(entries) == 2
    events = receipts(tmp_path)
    assert len(events) == 3
    assert {e.context.operation for e in events} == {
        UsageOperation.DOCUMENT_SUMMARY,
        UsageOperation.SUMMARY_EMBEDDING,
        UsageOperation.CHUNK_EMBEDDING,
    }
    assert sum(e.tokens.input_tokens for e in events) == 130
    summary = next(
        e
        for e in events
        if e.context.operation == UsageOperation.DOCUMENT_SUMMARY
    )
    assert summary.tokens.output_tokens == 20
    assert summary.tokens.cache_read_input_tokens == 30
    assert summary.response_model == "gpt-4o-mini-2024-07-18"
    assert summary.response_id == "response-fixture"
    assert all(e.context.document_id == context.document_id for e in events)
    assert all(e.context.owner_id == context.owner_id for e in events)
    assert len({e.event_id for e in events}) == 3
    assert {e.event_id for e in receipts(tmp_path)} == {
        e.event_id for e in events
    }
    serialized = "".join(p.read_text() for p in tmp_path.glob("*.json"))
    for secret in ("private", "secret-key", "forged-workspace"):
        assert secret not in serialized
    assert not list(tmp_path.glob("*.tmp"))
    assert all(
        p.stat().st_mode & 0o777 == 0o600 for p in tmp_path.glob("*.json")
    )


async def test_simple_workflow_failure_keeps_spend(service, context, tmp_path):
    """Keep completed model receipts when the real workflow fails at storage."""
    from fastapi import HTTPException

    async def parse_file(**kwargs):
        """Substitute file/database I/O with a known parsed chunk."""
        yield DocumentChunk(**chunk(context))

    async def fail_storage(entries):
        """Simulate the downstream database failing after model calls."""
        raise RuntimeError("storage unavailable")
        yield  # pragma: no cover

    service.parse_file = parse_file
    service.store_embeddings = fail_storage
    service.update_document_status = AsyncMock()
    workflow = simple_ingestion_factory(service)["ingest-files"]
    with pytest.raises(HTTPException, match="storage unavailable"):
        await workflow(
            {
                "document_id": str(context.document_id),
                "user": {
                    "id": str(context.owner_id),
                    "email": "test@example.com",
                },
                "metadata": {},
                "version": "v0",
                "size_in_bytes": 20,
                "ingestion_config": {},
                "file_data": {"filename": "fixture.txt"},
            }
        )
    events = receipts(tmp_path)
    assert len(events) == 3
    assert sum(e.tokens.input_tokens for e in events) == 120
    assert all(e.context.owner_id == context.owner_id for e in events)


async def test_concurrent_and_mixed_document_batches_keep_attribution(
    service, context, tmp_path
):
    """Never assign a mixed batch's full cost to its first document."""
    other = context.model_copy(
        update={"document_id": uuid4(), "owner_id": uuid4()}
    )

    async def ingest(items):
        """Consume the real concurrent embedding generator."""
        return [
            entry
            async for entry in service.embed_document(
                items, embedding_batch_size=2
            )
        ]

    results = await asyncio.gather(
        ingest([chunk(context), chunk(other)]), ingest([chunk(other)])
    )
    assert [len(r) for r in results] == [2, 1]
    events = receipts(tmp_path)
    assert len(events) == 3
    assert (
        sum(
            e.tokens.input_tokens
            for e in events
            if e.context.owner_id == context.owner_id
        )
        == 10
    )
    assert (
        sum(
            e.tokens.input_tokens
            for e in events
            if e.context.owner_id == other.owner_id
        )
        == 20
    )
    assert len({e.context.execution_id for e in events}) == 2
    await service.providers.embedding.async_get_embedding("unattributed query")
    assert len(receipts(tmp_path)) == 3


async def test_retry_after_invalid_vector_keeps_both_paid_calls(
    providers, context, tmp_path
):
    """Capture usage before vector extraction and retain each actual retry."""
    providers.embedding.litellm_aembedding.side_effect = [
        SimpleNamespace(
            model="embedding", usage=Usage(prompt_tokens=7), data=[{}]
        ),
        EmbeddingResponse(
            model="embedding",
            usage=Usage(prompt_tokens=11),
            data=[{"embedding": [1.0, 0.0]}],
        ),
    ]
    with usage_context(context):
        assert await providers.embedding.async_get_embedding("text") == [
            1.0,
            0.0,
        ]
    assert providers.embedding.litellm_aembedding.await_count == 2
    events = receipts(tmp_path)
    assert sorted(e.tokens.input_tokens for e in events) == [7, 11]
    assert len({e.event_id for e in events}) == 2
    assert {e.context.execution_id for e in events} == {context.execution_id}


async def test_short_embedding_response_cannot_cross_document_boundary(
    service, context, tmp_path
):
    """Reject missing vectors before another document's vectors can shift position."""
    other = context.model_copy(
        update={"document_id": uuid4(), "owner_id": uuid4()}
    )
    service.providers.embedding.litellm_aembedding.side_effect = None
    service.providers.embedding.litellm_aembedding.return_value = (
        EmbeddingResponse(
            model="embedding", usage=Usage(prompt_tokens=5), data=[]
        )
    )
    with pytest.raises(ValueError, match="Embedding count"):
        _ = [
            entry
            async for entry in service.embed_document(
                [chunk(context), chunk(other)]
            )
        ]
    service.providers.embedding.litellm_aembedding.assert_awaited_once()
    (event,) = receipts(tmp_path)
    assert event.context.document_id == context.document_id
    assert event.tokens.input_tokens == 5


async def test_failed_provider_calls_record_unknown_usage(
    providers, context, tmp_path
):
    """Do not turn network failures into zero-cost calls."""
    providers.embedding.litellm_aembedding.side_effect = RuntimeError(
        "private provider error"
    )
    with usage_context(context), pytest.raises(R2RException):
        await providers.embedding.async_get_embedding("text")
    events = receipts(tmp_path)
    assert len(events) == 2
    assert all(
        e.call_failed and e.usage_status == UsageStatus.MISSING for e in events
    )
    assert all(e.tokens.input_tokens is None for e in events)


async def test_failed_accounting_does_not_retry_model(
    providers, context, tmp_path, caplog
):
    """A failed local write must not cause additional provider spend."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("existing file")
    providers.embedding.config.app.usage_receipts_directory = str(blocked)
    with usage_context(context):
        result = await providers.embedding.async_get_embedding("text")
    assert result == [0.25, 0.75]
    assert providers.embedding.litellm_aembedding.await_count == 1
    assert "accounting gap" in caplog.text
    assert blocked.read_text() == "existing file"


def test_atomic_publication_failure_leaves_no_partial_receipt(
    context, tmp_path, monkeypatch, caplog
):
    """Do not expose incomplete JSON if flushing the temporary file fails."""
    from core.base import usage

    monkeypatch.setattr(
        usage.os, "fsync", Mock(side_effect=OSError("disk failure"))
    )
    with usage_context(context), capture_usage(str(tmp_path), "model") as call:
        call.response = {"usage": {"prompt_tokens": 2}}
    assert not list(tmp_path.iterdir())
    assert "accounting gap" in caplog.text


async def test_cancellation_records_gap_and_resets_context(context, tmp_path):
    """Cancellation remains visible without contaminating the next call."""
    with pytest.raises(asyncio.CancelledError):
        with usage_context(context), capture_usage(str(tmp_path), "model"):
            raise asyncio.CancelledError()
    with capture_usage(str(tmp_path), "unattributed") as call:
        call.response = {"usage": {"prompt_tokens": 5}}
    (event,) = receipts(tmp_path)
    assert event.call_failed
    assert event.usage_status == UsageStatus.MISSING


async def test_enrichment_and_reembedding_are_attributed(
    service, context, tmp_path
):
    """Cover the optional ingestion enrichment path and its second embedding."""
    item = chunk(context)
    item["text"] = item.pop("data")
    result = await service._get_enriched_chunk_text(
        chunk_idx=0,
        chunk=item,
        document_id=context.document_id,
        document_summary="summary",
        chunk_enrichment_settings=ChunkEnrichmentSettings(),
        list_document_chunks=[item],
    )
    assert result.text == "private summary"
    events = receipts(tmp_path)
    assert {e.context.operation for e in events} == {
        UsageOperation.CHUNK_ENRICHMENT,
        UsageOperation.ENRICHMENT_EMBEDDING,
    }
    assert len({e.context.execution_id for e in events}) == 1


@pytest.mark.parametrize("raw", [-1, True, 1.5, "12"])
def test_invalid_usage_is_not_coerced_or_priced(raw, context, tmp_path):
    """Reject malformed provider counters without aborting completed model work."""
    with usage_context(context), capture_usage(str(tmp_path), "model") as call:
        call.response = {"usage": {"prompt_tokens": raw}}
    (event,) = receipts(tmp_path)
    assert event.usage_status == UsageStatus.INVALID
    assert event.tokens.input_tokens is None


@pytest.mark.parametrize(
    "response", [None, {}, {"usage": None}, {"usage": {}}]
)
def test_missing_usage_is_unknown(response, context, tmp_path):
    """Preserve absent counters as null."""
    with usage_context(context), capture_usage(str(tmp_path), "model") as call:
        call.response = response
    (event,) = receipts(tmp_path)
    assert event.usage_status == UsageStatus.MISSING
    assert event.tokens.input_tokens is None


def test_anthropic_cache_subsets_are_not_added_twice(context, tmp_path):
    """Use LiteLLM's inclusive prompt count and retain private cache-write usage."""
    response = ModelResponse(
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=10,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=20,
        )
    )
    with (
        usage_context(context),
        capture_usage(str(tmp_path), "anthropic/claude") as call,
    ):
        call.response = response
    (event,) = receipts(tmp_path)
    assert event.tokens.model_dump() == {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_read_input_tokens": 30,
        "cache_write_input_tokens": 20,
    }


async def test_opt_in_and_disabled_summary(service, context, tmp_path):
    """Defaults leave ingestion behavior and the filesystem unchanged."""
    service.config.ingestion.skip_document_summary = True
    await service.augment_document_info(document(context), [chunk(context)])
    service.providers.llm.acompletion.assert_not_awaited()
    service.providers.embedding.config.app.usage_receipts_directory = None
    assert (
        len([e async for e in service.embed_document([chunk(context)])]) == 1
    )
    assert receipts(tmp_path) == []


def test_config_roundtrip(tmp_path):
    """Load opt-in settings through native R2R TOML configuration."""
    config = R2RConfig({"app": {"usage_receipts_directory": str(tmp_path)}})
    restored = R2RConfig(toml.loads(config.to_toml()))
    assert restored.completion.app.usage_receipts_directory == str(tmp_path)
    assert restored.embedding.app.usage_receipts_directory == str(tmp_path)
    assert R2RConfig({}).app.usage_receipts_directory is None


def test_sync_providers(providers, context, tmp_path):
    """Synchronous LiteLLM boundaries use the same receipt contract."""
    providers.embedding.litellm_embedding = Mock(
        return_value=EmbeddingResponse(
            model="embedding",
            usage=Usage(prompt_tokens=9),
            data=[{"embedding": [1.0, 0.0]}],
        )
    )
    providers.llm.completion = Mock(
        return_value=ModelResponse(usage=Usage(prompt_tokens=7))
    )
    with usage_context(context):
        assert providers.embedding.get_embedding("text") == [1.0, 0.0]
        providers.llm._execute_task_sync(
            {
                "messages": [],
                "generation_config": GenerationConfig(
                    model="openai/gpt-4o-mini"
                ),
                "kwargs": {},
            }
        )
    assert sorted(e.tokens.input_tokens for e in receipts(tmp_path)) == [7, 9]


@pytest.mark.parametrize(
    "raw_usage, expected, status",
    [
        ({"prompt_tokens": 23, "total_tokens": 23}, 23, UsageStatus.OBSERVED),
        ({"prompt_tokens": 0, "total_tokens": 0}, 0, UsageStatus.OBSERVED),
        (None, None, UsageStatus.MISSING),
        ({}, None, UsageStatus.MISSING),
        ({"prompt_tokens": -1, "total_tokens": -1}, None, UsageStatus.INVALID),
    ],
)
async def test_real_litellm_http_embedding(
    providers, context, tmp_path, raw_usage, expected, status
):
    """Validate actual LiteLLM/OpenAI response conversion with only HTTP simulated."""
    import litellm

    def respond(request):
        """Return a provider-shaped embedding response with known usage."""
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2],
                    }
                ],
                "usage": raw_usage,
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as http:
        client = AsyncOpenAI(
            api_key="fixture-key",
            http_client=http,
            base_url="https://fixture.invalid/v1",
        )
        providers.embedding.litellm_aembedding = litellm.aembedding
        with usage_context(context):
            result = await providers.embedding.async_get_embedding(
                "hello", client=client
            )
    assert result == [0.1, 0.2]
    (event,) = receipts(tmp_path)
    assert event.tokens.input_tokens == expected
    assert event.tokens.output_tokens is None
    assert event.usage_status == status
    assert event.usage_source == UsageSource.PROVIDER_RESPONSE
    assert event.response_model == "text-embedding-3-small"
    await asyncio.sleep(0.05)


async def test_real_litellm_http_summary_retry(providers, context, tmp_path):
    """Observe R2R retries separately with the real LiteLLM/OpenAI adapters."""
    import litellm

    attempts = 0
    existing_logger = Mock()

    def respond(request):
        """Fail the first HTTP call and return a priced completion on retry."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                500, json={"error": {"message": "try again"}}
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini-2024-07-18",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "summary"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 32,
                    "completion_tokens": 6,
                    "total_tokens": 38,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as http:
        client = AsyncOpenAI(
            api_key="fixture-key",
            http_client=http,
            base_url="https://fixture.invalid/v1",
        )
        providers.llm.acompletion = litellm.acompletion
        with usage_context(context):
            response = await providers.llm.aget_completion(
                messages=[{"role": "user", "content": "summarize"}],
                generation_config=GenerationConfig(model="openai/gpt-4o-mini"),
                client=client,
                logger_fn=existing_logger,
            )
    assert response.choices[0].message.content == "summary"
    assert attempts == 2
    events = receipts(tmp_path)
    assert len(events) == 2
    assert sum(e.call_failed for e in events) == 1
    observed = next(e for e in events if not e.call_failed)
    assert observed.tokens.input_tokens == 32
    assert observed.tokens.output_tokens == 6
    assert observed.tokens.cache_read_input_tokens == 8
    assert observed.response_id == "chatcmpl-fixture"
    assert observed.provider == "openai"
    assert existing_logger.call_count > 0
    await asyncio.sleep(0.05)


async def test_real_embedding_retry_and_concurrent_raw_usage(
    providers, context, tmp_path
):
    """Keep raw counters task-local across SDK threads, concurrency and retries."""
    import litellm

    other = context.model_copy(
        update={"document_id": uuid4(), "owner_id": uuid4()}
    )
    attempts = {"first": 0, "second": 0}

    async def respond(request):
        """Interleave requests and fail one HTTP attempt before returning usage."""
        name = json.loads(request.content)["input"][0]
        attempts[name] += 1
        await asyncio.sleep(0)
        if name == "first" and attempts[name] == 1:
            return httpx.Response(500, json={"error": {"message": "retry"}})
        count = 7 if name == "first" else 11
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2],
                    }
                ],
                "usage": {"prompt_tokens": count, "total_tokens": count},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as http:
        client = AsyncOpenAI(
            api_key="fixture-key",
            http_client=http,
            base_url="https://fixture.invalid/v1",
        )
        providers.embedding.litellm_aembedding = litellm.aembedding

        async def embed(name, attribution):
            """Bind each concurrent request to its own document."""
            with usage_context(attribution):
                return await providers.embedding.async_get_embedding(
                    name, client=client
                )

        assert (
            await asyncio.gather(
                embed("first", context), embed("second", other)
            )
            == [[0.1, 0.2]] * 2
        )
    assert attempts == {"first": 2, "second": 1}
    events = receipts(tmp_path)
    assert len(events) == 3
    observed = [e for e in events if not e.call_failed]
    assert {e.context.owner_id: e.tokens.input_tokens for e in observed} == {
        context.owner_id: 7,
        other.owner_id: 11,
    }
    assert all(
        e.usage_source == UsageSource.PROVIDER_RESPONSE for e in observed
    )
    await asyncio.sleep(0.05)
