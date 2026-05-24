# censAI

![censAI](./design/image.png)

A censorship program to let you clean shows, and make any show watchable for any audiance.

This was made because I wanted to watch shows with people who were not rated for those shows. Some shows have stories worth watching, but you need to censor either some words, or some scenes to make it suitable for a wider audience.

## Working


## What you need
1. The show downloaded to a folder
2. Subtitles of that show in the same folder for its episode. (synced properly)
3. Ollama up and running, or `CENSAI_USE_LLM_GATEWAY=true` with Gateway env configured

## Todo
1. Implement for shows
2. Implement for movies
3. Better ui.

## Caching

All NudeNet and LLM calls are cached in a SQLite database located at
`<media-folder>/temp/censai.sqlite`.

- Vision LLM calls are keyed by `(image_sha256, model)`, with a
  `(perceptual_hash, model)` -> `image_sha256` alias so near-identical
  frames inside one scene hit the cache for free.
- NudeNet calls are keyed by `image_sha256`.
- Subtitle profanity rewrites are keyed by `(text_hash, model)`.

If you change a prompt or want to re-run from scratch, delete
`temp/censai.sqlite`. The intermediate scenedetect CSV and snapshot images
stay around -- only the model outputs are recomputed.

## Configuration

Most non-secret runtime settings are now stored in the main config DB
(`censai_configs`) inside Postgres or the root SQLite fallback. The env values
below act as bootstrap defaults for first run and remain the fallback when the
central DB is unavailable. Secrets and DB connection details still stay in env.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CENSAI_USE_LLM_GATEWAY` | `false` | When true, all LLM calls go through LLM Gateway |
| `LLM_GATEWAY_URL` | `https://llmgateway.krishnarajthadesar.in` | Gateway base URL |
| `LLM_GATEWAY_API_KEY` | empty | Gateway API key sent as `X-API-Key` |
| `LLM_GATEWAY_CHAT_PATH` | `/api/chat` | Native Gateway chat endpoint |
| `LLM_GATEWAY_VISION_MODEL` | `gemma4:27b` | Gateway model for per-frame classification |
| `LLM_GATEWAY_PROFANITY_MODEL` | `gemma4:27b` | Gateway model for subtitle rewriting |
| `CENSAI_LLM_GATEWAY_MAX_PARALLEL_CALLS` | `3` | Max concurrent Gateway calls for uncached vision/profanity work |
| `CENSAI_OLLAMA_MAX_PARALLEL_CALLS` | `1` | Max concurrent local Ollama calls for uncached vision/profanity work |
| `CENSAI_VISION_MODEL` | `qwen3-vl:4b` | Local Ollama model for per-frame classification |
| `CENSAI_PROFANITY_MODEL` | `mistral` | Local Ollama model for subtitle rewriting |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DATABASE_URL` | empty | Optional Postgres SQLAlchemy URL for the main tracking DB |
| `CENSAI_MAIN_SQLITE_PATH` | `censai.sqlite3` | Root-level SQLite fallback when `DATABASE_URL` is blank |
| `DB_SCHEMA` | `censai` | Postgres schema for tracking tables |
| `CENSAI_RETRY_DELAY_HOURS` | `24` | Rate-limit retry fallback window |
| `CENSAI_MEDIA_FOLDER` | `/media` | Folder scanned by the long-running pod worker |

For local Ollama, start with `CENSAI_OLLAMA_MAX_PARALLEL_CALLS=2` and increase
only if the machine has enough GPU/RAM headroom. CensAI will submit concurrent
requests, but actual model execution also depends on the Ollama server's own
parallelism and loaded-model settings.

## Central Tracking and UI

Local SQLite caches stay in each media folder. CensAI also keeps a main
tracking DB for detected videos, status, retry timing, and editable config
values. If `DATABASE_URL` is set, that DB is Postgres under `DB_SCHEMA`;
otherwise it defaults to root-level SQLite at `sqlite:///censai.sqlite3`.

Run `python main.py` to start the same-pod developer UI on port `8000`. The UI
shows detected videos, whether they have been censored, subtitle availability,
attempts, errors, retry times, and model/runtime details.

After the first successful startup, edit non-secret runtime values from the UI
or directly in `censai_configs`; changing `.env` alone will not overwrite
existing central config rows.

Scanning only discovers videos. The worker processes videos only after they are
queued from the UI, either individually, in a selected batch, or recursively by
folder/subfolder. Rate-limited videos that were already queued are retried after
the configured fallback window.

The `k3s/` folder contains ArgoCD/kustomize manifests for the long-running pod.
Runtime secrets are pulled from Vault through External Secrets, matching the
pattern used by the other homelab apps.

# Running the project
