# censAI

![censAI](./design/image.png)

A censorship program to let you clean shows, and make any show watchable for any audiance.

This was made because I wanted to watch shows with people who were not rated for those shows. Some shows have stories worth watching, but you need to censor either some words, or some scenes to make it suitable for a wider audience.

## Working


## What you need
1. The show downloaded to a folder
2. Subtitles of that show in the same folder for its episode. (synced properly)
3. Ollama up and running

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

| Variable                 | Default              | Purpose                                  |
| ------------------------ | -------------------- | ---------------------------------------- |
| `CENSAI_VISION_MODEL`    | `qwen3-vl:4b`        | Ollama model for per-frame classification |
| `CENSAI_PROFANITY_MODEL` | `mistral`            | Ollama model for subtitle rewriting       |
| `OLLAMA_HOST`            | `http://localhost:11434` | Ollama server URL                     |

# Running the project
