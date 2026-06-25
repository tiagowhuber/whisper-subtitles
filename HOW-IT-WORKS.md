# How whisper-subtitles works

The whole pipeline in one sentence:

> **Whisper produces `{time, English text}` pairs; only the text is sent to GPT
> for translation; the Spanish text comes back and is slotted onto Whisper's
> original timestamps by position; the SRT is just those `{time, Spanish text}`
> pairs written in subtitle format.** Timestamps are computed once, locally, and
> never change.

The detailed flow, using a concrete example.

## 1. What Whisper returns

`model.transcribe(path)` decodes the audio (via ffmpeg) and returns a big dict.
The important part is `result["segments"]` — a list, one entry per chunk of
speech, each with start/end times **and** the text. A raw segment looks like:

```python
{"id": 3, "start": 6.16, "end": 7.38, "text": " He's just a memory of mine.",
 "tokens": [...], "avg_logprob": -0.21, "no_speech_prob": 0.01, ...}
```

It has extra fields (token IDs, confidence scores) that we don't need.

## 2. What's held in memory

`transcribe()` keeps only the three fields that matter and strips whitespace:

```python
segments = [
  {"start": 0.00, "end": 4.30, "text": "Look at this version of you."},
  {"start": 6.16, "end": 7.38, "text": "He's just a memory of mine."},
  # ... ~641 of these for an episode ...
]
```

A plain list of `{start, end, text}` dicts. **The timestamps live here and never
go anywhere else.**

## 3. What's sent to GPT

In `_translate_batch()`, a batch of 40 segments becomes a JSON object keyed by
**position only** — no timestamps:

```python
payload = {
  "0": "Look at this version of you.",
  "1": "He's just a memory of mine.",
  # ...
}
```

That JSON is the user message; the system message is the translator instruction
("translate each numbered line to neutral Latin-American Spanish, return the same
keys"). GPT is asked to reply as JSON (`response_format={"type":"json_object"}`),
so it returns:

```python
{"0": "Mira esta versión de ti.", "1": "Él es solo un recuerdo mío.", ...}
```

**Only the text strings cross the network.** GPT never sees timing — it just
translates 40 numbered lines and hands back 40 numbered translations.

## 4. Marrying it back (the key trick)

Because the keys are just positions, the code reads them back in order and
overwrites the text **in place**, leaving each segment's `start`/`end` untouched:

```python
out = [data.get(str(i)) for i in range(len(texts))]   # ["Mira esta versión...", ...]
if any(v is None for v in out): raise ValueError(...)  # safety: every line came back
# ...
segments[start + off]["text"] = es                     # English text -> Spanish text
```

That `if any(... is None)` check is the safety net: if GPT dropped or renumbered
a line, the batch is rejected and retried one segment at a time, so a translation
can never end up attached to the wrong timestamp.

After this step the in-memory list is the same shape as before — just Spanish:

```python
{"start": 0.00, "end": 4.30, "text": "Mira esta versión de ti."}
```

## 5. Building the SRT

`write_srt()` walks that list and prints the standard SRT block per cue: a
running number, the timing line, the text, a blank line.

```python
index += 1
f.write(f"{index}\n")
f.write(f"{format_ts(seg['start'])} --> {format_ts(seg['end'])}\n")
f.write(f"{text}\n\n")
```

`format_ts(6.16)` converts seconds (a float) into the SRT clock format
`00:00:06,160` — hours/minutes/seconds with a comma before milliseconds. Result:

```
2
00:00:06,160 --> 00:00:07,380
Él es solo un recuerdo mío.
```
