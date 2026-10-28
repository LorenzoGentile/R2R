# Ingestion usage receipts

This fork can persist content-free model usage from the LiteLLM ingestion
providers. The feature is disabled by default. It does not change document API
responses, create a billing ledger, or estimate tokens from document size.

## Enable

Use an operator-controlled R2R TOML configuration:

```toml
[app]
usage_receipts_directory = "/data/r2r-usage"

[completion]
provider = "litellm"

[embedding]
provider = "litellm"
```

Keep your existing model and other configuration settings. Mount the receipt
directory on durable storage in every R2R process performing ingestion, and make
it available to your trusted accounting importer. The directory is created with
mode `0700`; existing directory permissions are not changed. Published files use
mode `0600`. Do not expose this directory through a public file server.

There is no new environment variable, database migration or HTTP endpoint.
The settings belong to the server configuration, never to a document's metadata
or an ingestion request. Existing deployments continue to work without enabling
the feature.

## Coverage

The shared ingestion service supplies task-local attribution for:

- document summary completion and summary embedding;
- chunk embedding, including chunk uploads and updates;
- optional chunk enrichment completion and re-embedding.

Both simple and Hatchet workflows call these service methods. Concurrent jobs
cannot overwrite each other's context. Mixed-document embedding batches are
split into homogeneous calls, so a batch is never charged entirely to its first
document. Other batching and provider return values stay the same.

Receipts are captured before summary/vector validation and document persistence.
Successful provider calls remain recorded if a later step fails. Each R2R retry
gets its own event ID. With recording enabled, nested LiteLLM/OpenAI SDK retries
are disabled; retry policy remains with R2R so each attempt is observable.
LiteLLM 1.69.3's OpenAI embedding adapter replaces `max_retries=0` with its
default, so the embedding boundary uses `-1` (no remaining SDK retries) instead.
The HTTP tests cover this compatibility workaround; requalify it on SDK upgrades.
Transport failures and cancellations produce receipts
with unknown counters unless usage was received.

Unattributed calls, query embedding, chat, graph extraction, media parsing, other
provider adapters and streaming usage are outside this ingestion contract. This
is not a complete account of every model call an R2R deployment can make.

## Receipt contract

The authoritative Pydantic schema is `core.base.usage.UsageReceipt`, version `1`.
Every completed provider attempt publishes one `<event_id>.json` file containing:

- a UUID event ID and UTC call-start timestamp (`occurred_at`);
- R2R document, owner and available collection IDs, plus an operation;
- an `execution_id` grouping one service-stage invocation and its retries;
- the adapter, available provider identity, requested and returned model names,
  and available response ID;
- whether the provider call failed, the usage source and counter status;
- nullable input, output, cache-read and cache-write token counts.

A summary and its embedding share a stage execution ID. Chunk embedding batches
share another. Enrichment and its re-embedding share an ID per chunk. A stage ID
is **not** a whole-workflow task ID or an idempotency key. `call_failed` describes
the provider call, not later validation or final document status.

`input_tokens` includes cached input. Cache-read and cache-write counts are
subsets; do not add them to input again. Missing counters remain `null`.
Negative, fractional, boolean, string and inconsistent cache counters are
rejected, and the receipt is marked `invalid`. No price is calculated here.

For `openai/` routes (including OpenAI-compatible gateways), a synchronous
LiteLLM post-call hook reads the provider response before LiteLLM can replace
missing counters with zeros. These receipts have
`usage_source="provider_response"`; an actual zero remains distinct from missing
usage. The hook retains only validated counters, not raw responses or prompts.
It uses task-local state, preserves existing callbacks and does not install a
per-request global callback.

Other LiteLLM routes use `usage_source="litellm"`. Their counters reflect SDK
normalization, including inclusive Anthropic input/cache semantics. Some SDKs
synthesize zeros for absent usage; qualify their raw-response mapping before
treating these receipts as authoritative billing data. Returned model/provider
identity can be absent; the requested model is retained separately, without
claiming it was the actual model served.

No text, vectors, arbitrary metadata, credentials, endpoint URLs or provider
error bodies are written. Workspace/Project identity is deliberately absent:
an importer must resolve the document against its own trusted ownership records.
Collection IDs may be empty before ingestion assigns collections, and must not
be treated as authorization claims. Use that trusted mapping to distinguish
source documents (L1) from wiki pages (L2).

## Durability and import

Files are flushed and fsynced before atomic publication; the containing
directory is fsynced too. Import only `*.json`, validate the schema, and enforce a
unique constraint on `event_id` in your ledger. Replaying a file must not charge
it again. Archive/remove receipts only after the importer commits its ledger
transaction. R2R never deletes receipts or reruns model work to replay them.

Receipt failures emit a `CRITICAL` accounting-gap log containing the event ID,
exception type and available system error code (for example, `ENOSPC` or
`EACCES`). Exception messages and chains are omitted because they can contain
provider response bodies. These failures do not fail or retry an already paid
model call. Monitor these logs,
`missing`/`invalid` receipts, storage capacity and importer backlog. A disk failure,
process kill before publication, or a provider charge whose response never
arrives cannot be reconstructed from these files. This is not exact invoice
reconciliation. Small synchronous filesystem flushes add I/O to each recorded
call; use suitable local persistent storage.

## Validation

Run from `py/` with core and development dependencies installed:

```sh
python -m pytest tests/unit/usage/test_receipts.py -q -o addopts=''
```

Tests exercise real ingestion/provider code with database/file boundaries
simulated, plus actual LiteLLM/OpenAI HTTP conversion with an HTTP mock transport.
They cover known usage, missing versus zero counters, malformed usage, cache
semantics, retries, concurrent and mixed-document work, downstream failure,
cancellation, write failure, replay identities, file permissions and disabled
configuration. No paid model calls or live database are needed.

The repository's pre-existing `uv.lock` pins LiteLLM 1.64.1, older than the
`pyproject.toml` minimum of 1.69.3. Validation for this change uses LiteLLM 1.69.3
and Python 3.12; when using `uv sync --frozen`, update LiteLLM in the test
environment to that version. This change does not refresh unrelated dependencies.
