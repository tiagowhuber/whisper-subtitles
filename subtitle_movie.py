#!/usr/bin/env python3
"""
Movie → Spanish SRT subtitles.

Pipeline:
  1. Whisper transcribes the movie's audio in its ORIGINAL language (with
     per-segment timestamps). Whisper decodes the video via ffmpeg internally.
  2. Each segment's text is translated to Spanish with OpenAI (gpt-5-mini by
     default). Timestamps from Whisper are never altered — only the text.
  3. A UTF-8 .srt file is written next to the movie so VLC auto-loads it.

Usage:
  OPENAI_API_KEY=...  python subtitle_movie.py movie.mkv
  python subtitle_movie.py movie.mkv -o out.es.srt --whisper-model medium
  python subtitle_movie.py movie.mkv --source-lang en --openai-model gpt-5-mini
  python subtitle_movie.py movie.mkv --no-translate     # source-language SRT only
"""

import argparse
import json
import logging
import os
import re
import sys
import time

# This machine's AMD GPU needs ROCm spoofed to gfx10.3.0 (same as the
# whisper-daemon systemd service). Set before torch is ever imported.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large")  # tiny/base/small/medium/large
OPENAI_MODEL = "gpt-5-mini"
BATCH_SIZE = 40        # segments per translation request
MAX_RETRIES = 4        # OpenAI transient-error retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("subtitle_movie")


# ── Whisper ──────────────────────────────────────────────────────────────────

def load_model(model_name: str, want_device: str = "auto"):
    """Load Whisper. want_device is 'auto', 'cpu', or 'cuda' (mirrors whisper_daemon.py)."""
    import torch
    import whisper

    if want_device == "cpu":
        device = "cpu"
    elif want_device == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading whisper model '%s' on %s …", model_name, device)
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    try:
        model = whisper.load_model(model_name, device=device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device == "cuda":
            log.warning("GPU load failed (%s). The whisper daemon may be holding "
                        "VRAM — retrying on CPU. (Stop the daemon, or use a smaller "
                        "--whisper-model, for GPU speed.)", e)
            device = "cpu"
            model = whisper.load_model(model_name, device=device)
        else:
            raise
    log.info("Model loaded.")
    return model, device


def transcribe(model, device, path: str, source_lang, start_min=0.0, dur_min=None):
    """Transcribe the movie, keeping the source language. Returns Whisper segments.

    Only the window [start_min, start_min + dur_min) minutes of audio is
    transcribed: the audio is sliced before Whisper sees it, so the rest of the
    film is never processed. dur_min=None means "to the end". Segment timestamps
    are shifted by start_min so they line up with the real movie time.
    """
    import whisper

    offset = (start_min or 0.0) * 60.0
    audio = path
    if start_min or dur_min:
        # Whisper decodes to a 16 kHz mono float array; keep only our window.
        full = whisper.load_audio(path)
        sr = whisper.audio.SAMPLE_RATE
        lo = int((start_min or 0.0) * 60 * sr)
        hi = int(((start_min or 0.0) + dur_min) * 60 * sr) if dur_min else None
        audio = full[lo:hi]
        log.info("Transcribing window: minute %g → %s (%d of %d audio samples).",
                 start_min or 0.0, f"{(start_min or 0.0) + dur_min:g}" if dur_min else "end",
                 len(audio), len(full))

    log.info("Transcribing '%s' (language=%s) …", path, source_lang or "auto")
    result = model.transcribe(
        audio,
        task="transcribe",                 # keep SOURCE language, not English
        language=source_lang,              # None → auto-detect
        fp16=(device == "cuda"),
        verbose=False,
    )
    segments = [
        {"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"].strip()}
        for s in result["segments"]
    ]
    log.info("Detected language: %s | %d segments", result.get("language"), len(segments))
    return segments


# ── Translation (OpenAI) ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional subtitle translator. Translate each numbered "
    "subtitle line into neutral Latin American Spanish. Keep each translation "
    "concise and natural for on-screen reading, preserving meaning, tone, and "
    "register. Do not merge, split, reorder, or drop lines. Return ONLY a JSON "
    "object that maps each input key to its translated Spanish string, with the "
    "exact same set of keys."
)


def _translate_batch(client, model: str, texts: list) -> list:
    """Translate one list of strings → list of Spanish strings (same length).

    Raises ValueError if the model's response can't be aligned 1:1.
    """
    payload = {str(i): t for i, t in enumerate(texts)}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            out = [data.get(str(i)) for i in range(len(texts))]
            if any(v is None for v in out):
                raise ValueError("response missing one or more keys")
            return [str(v) for v in out]
        except Exception as e:  # transient API errors or alignment failures
            last_err = e
            if attempt < MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                log.warning("translate attempt %d failed (%s); retrying in %ss",
                            attempt, e, wait)
                time.sleep(wait)
    raise ValueError(f"batch translation failed after {MAX_RETRIES} attempts: {last_err}")


def translate_segments(segments: list, model: str, batch_size: int) -> None:
    """Translate every segment's text to Spanish in place."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY
    texts = [s["text"] for s in segments]
    total = (len(texts) + batch_size - 1) // batch_size
    log.info("Translating %d segments to Spanish via %s (%d batches) …",
             len(texts), model, total)

    for b, start in enumerate(range(0, len(texts), batch_size), 1):
        chunk = texts[start:start + batch_size]
        try:
            translated = _translate_batch(client, model, chunk)
        except ValueError as e:
            # Fall back to one-at-a-time so a single bad batch can't desync timing.
            log.warning("batch %d/%d failed (%s); falling back to per-segment",
                        b, total, e)
            translated = []
            for t in chunk:
                try:
                    translated.append(_translate_batch(client, model, [t])[0])
                except ValueError:
                    log.error("segment translation failed; keeping original text")
                    translated.append(t)
        for off, es in enumerate(translated):
            segments[start + off]["text"] = es
        log.info("  batch %d/%d translated", b, total)


# ── SRT output ───────────────────────────────────────────────────────────────

def format_ts(seconds: float) -> str:
    """Seconds → 'HH:MM:SS,mmm'."""
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list, out_path: str) -> None:
    segments = sorted(segments, key=lambda s: s["start"])
    index = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            index += 1
            f.write(f"{index}\n")
            f.write(f"{format_ts(seg['start'])} --> {format_ts(seg['end'])}\n")
            f.write(f"{text}\n\n")
    log.info("Wrote %s (%d cues)", out_path, index)


_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _ts_to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def read_srt(path: str) -> list:
    """Parse an existing .srt back into [{'start','end','text'}] (best effort)."""
    segs = []
    with open(path, encoding="utf-8-sig") as f:
        content = f.read()
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        ts_line = next((ln for ln in lines if _TS.search(ln)), None)
        if not ts_line:
            continue
        m = _TS.search(ts_line)
        text = "\n".join(lines[lines.index(ts_line) + 1:]).strip()
        if not text:
            continue
        segs.append({
            "start": _ts_to_seconds(*m.group(1, 2, 3, 4)),
            "end": _ts_to_seconds(*m.group(5, 6, 7, 8)),
            "text": text,
        })
    return segs


def merge_window(out_path: str, new_segments: list, win_start: float, win_end) -> list:
    """Splice freshly-transcribed cues into an existing SRT.

    Existing cues whose start falls inside the just-transcribed window
    [win_start, win_end) are dropped (this run is the source of truth for that
    span); everything else is kept. Returns the combined, time-sorted list.
    win_end=None means "to the end of the film".
    """
    existing = read_srt(out_path) if os.path.isfile(out_path) else []
    kept = [
        c for c in existing
        if c["start"] < win_start or (win_end is not None and c["start"] >= win_end)
    ]
    if existing:
        log.info("Merging into existing %s: kept %d cue(s) outside the window, "
                 "added %d new.", os.path.basename(out_path), len(kept), len(new_segments))
    return kept + new_segments


# ── Pipeline ─────────────────────────────────────────────────────────────────

def process_window(model, device, args, out_path: str,
                   start_min: float, dur_min) -> int:
    """Transcribe + translate one [start_min, start_min+dur_min) window and splice
    it into out_path. Returns the number of cues produced for this window."""
    segments = transcribe(model, device, args.input, args.source_lang, start_min, dur_min)
    if not segments:
        log.warning("No speech segments in this window.")
        return 0

    if args.no_translate:
        log.info("--no-translate set; writing source-language SRT.")
    else:
        translate_segments(segments, args.openai_model, args.batch_size)

    win_start = start_min * 60.0
    win_end = (start_min + dur_min) * 60.0 if dur_min else None
    merged = merge_window(out_path, segments, win_start, win_end)
    write_srt(merged, out_path)
    return len(segments)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Generate Spanish SRT subtitles for a movie.")
    p.add_argument("input", help="Path to the movie file")
    p.add_argument("-o", "--output", help="Output .srt path (default: <movie>.es.srt)")
    p.add_argument("--whisper-model", default=WHISPER_MODEL,
                   help=f"Whisper model size (default: {WHISPER_MODEL})")
    p.add_argument("--source-lang", default=None,
                   help="Source audio language code (default: auto-detect)")
    p.add_argument("--from", dest="start", type=float, default=None, metavar="N",
                   help="Start at minute N (default: 0). Cues are merged into an "
                        "existing .es.srt, so you can do the rest of a film in a second pass.")
    p.add_argument("--minutes", type=float, default=None, metavar="N",
                   help="Transcribe only N minutes (the window length). "
                        "With --from this is the span starting there; alone it's the "
                        "first N minutes (default: to the end of the film).")
    p.add_argument("--head", type=float, default=None, metavar="N",
                   help="Two-stage: subtitle the first N minutes and write the SRT so "
                        "you can start watching, then continue to the end in the same "
                        "run, merging as it goes. (Conflicts with --from/--minutes.)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Where to run Whisper (default: auto; falls back to CPU on GPU OOM)")
    p.add_argument("--openai-model", default=OPENAI_MODEL,
                   help=f"OpenAI model for translation (default: {OPENAI_MODEL})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"Segments per translation request (default: {BATCH_SIZE})")
    p.add_argument("--no-translate", action="store_true",
                   help="Skip translation; emit a source-language SRT (debug)")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        log.error("Input file not found: %s", args.input)
        return 1

    if args.minutes is not None and args.minutes <= 0:
        log.error("--minutes must be a positive number.")
        return 1
    if args.start is not None and args.start < 0:
        log.error("--from must be zero or a positive number.")
        return 1
    if args.head is not None and args.head <= 0:
        log.error("--head must be a positive number.")
        return 1
    if args.head is not None and (args.start is not None or args.minutes is not None):
        log.error("--head can't be combined with --from/--minutes "
                  "(it does both stages itself).")
        return 1

    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(args.input)
        out_path = base + (".srt" if args.no_translate else ".es.srt")

    if not args.no_translate and not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY is not set (needed for translation). "
                  "Set it, or pass --no-translate for a source-language SRT.")
        return 1

    model, device = load_model(args.whisper_model, args.device)

    if args.head is not None:
        # Stage 1: first N minutes → write the file so the user can start watching.
        log.info("Stage 1/2: first %g minute(s) — write, then keep going.", args.head)
        n1 = process_window(model, device, args, out_path, 0.0, args.head)
        log.info("First %g minute(s) ready in '%s' — you can start watching. "
                 "Continuing with the rest…", args.head, os.path.basename(out_path))
        # Stage 2: minute N → end, merged into the same file.
        log.info("Stage 2/2: minute %g → end.", args.head)
        n2 = process_window(model, device, args, out_path, args.head, None)
        if n1 + n2 == 0:
            log.error("No speech segments found.")
            return 1
    else:
        start_min = args.start or 0.0
        n = process_window(model, device, args, out_path, start_min, args.minutes)
        if n == 0:
            log.error("No speech segments found.")
            return 1

    log.info("Done. Open the movie in VLC — '%s' loads automatically.",
             os.path.basename(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
