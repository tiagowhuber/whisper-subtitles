# whisper-subtitles

Generate **Spanish `.srt` subtitles** for a movie from its (non-Spanish) audio,
so you can watch it with Spanish subtitles in VLC. It only ever writes a `.srt`
text file next to the video — it never modifies or re-encodes the video.

How it works (full walkthrough in [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md)):

1. **OpenAI Whisper** transcribes the audio in its original language, with
   per-segment timestamps (Whisper decodes the video via `ffmpeg`).
2. **OpenAI `gpt-5-mini`** translates each subtitle line to Spanish. Whisper's
   timestamps are kept exactly — only the text is translated, so sync is preserved.
3. A UTF-8 `.srt` is written next to the movie, so VLC loads it automatically.

> Why a translation step? Whisper can *translate*, but only **to English** —
> never to Spanish. So we transcribe in the source language and translate the
> text with an LLM.

## Requirements

- `ffmpeg` on `PATH`.
- Python with `torch`, `openai-whisper`, and `openai` installed (a GPU is
  strongly recommended for the `large` model; CPU works but is slow).
- An OpenAI API key.

## Quick start

```sh
./install.sh                              # links the command + seeds the config
$EDITOR ~/.config/whisper-subtitles.env   # paste your OPENAI_API_KEY
# open a new terminal, then:
whisper-subtitles '/path/to/movie.mkv'
```

`whisper-subtitles` writes `movie.es.srt` next to the video. Open the video in
VLC and the subtitles load automatically (or **Subtitle → Add Subtitle File**).

> Tip: type `whisper-subtitles ` then **Tab-complete** the filename — the
> installed zsh alias also lets unquoted paths with spaces and `[brackets]` work.

## Direct usage (more options)

The wrapper just manages the GPU daemon and your key; the real tool is
`subtitle_movie.py`:

```sh
PY=python3   # or ~/whisper-service/venv/bin/python

$PY subtitle_movie.py movie.mkv                       # -> movie.es.srt
$PY subtitle_movie.py movie.mkv -o out.es.srt
$PY subtitle_movie.py movie.mkv --whisper-model medium   # faster, less accurate
$PY subtitle_movie.py movie.mkv --source-lang en         # skip language auto-detect
$PY subtitle_movie.py movie.mkv --device cpu             # force CPU
$PY subtitle_movie.py movie.mkv --no-translate           # source-language SRT, no API key
```

## Configuration (environment variables)

The `whisper-subtitles` wrapper reads these (also accepted in
`~/.config/whisper-subtitles.env`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — (required) | Your OpenAI key |
| `WHISPER_SUBTITLES_PYTHON` | venv if present, else `python3` | Interpreter to run with |
| `WHISPER_SUBTITLES_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `WHISPER_SUBTITLES_DAEMON` | `whisper-daemon` | systemd `--user` unit to pause during the run; set empty to disable |

## GPU notes

- On **NVIDIA**, CUDA works out of the box if your `torch` build has CUDA.
- On **AMD (ROCm)**, some cards need `HSA_OVERRIDE_GFX_VERSION` set. This repo
  defaults it to `10.3.0` (RDNA2, e.g. RX 6700 XT) via `os.environ.setdefault`
  in `subtitle_movie.py` — override it with your card's value if different, or
  unset it on NVIDIA (the `setdefault` won't clobber an existing value).
- If the GPU runs out of memory, the tool automatically falls back to CPU.

## This author's setup

On the original machine this **reuses an existing `whisper-service` venv**
(already has `torch` + `openai-whisper`) and pauses that project's
`whisper-daemon` (voice-to-text) to free the GPU during a run, then restarts it.
On a machine without that daemon, `WHISPER_SUBTITLES_DAEMON` simply finds nothing
to pause and the step is skipped.

## Notes

- `--no-translate` verifies transcription/timestamps without using the API.
- Translation is batched (`--batch-size`, default 40) for context and speed; if a
  batch fails to align 1:1 it retries per-segment, so timing never desyncs.
- `--whisper-model large` (default) is the accuracy sweet spot on GPU.
