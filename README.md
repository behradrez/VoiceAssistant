# Voice Assistant (Offline)

This project includes a local/offline wake-word listener and speech transcription loop.

## What it does

- Continuously listens for one or more activation terms (default: `listen bot`)
- After activation, captures the next spoken instruction
- Stores the latest instruction in memory (`latest_instruction`) and prints it

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Download a Vosk model and unpack it locally, for example:
   - `vosk-model-small-en-us-0.15`

## Run

```bash
python3 local_voice_assistant.py --model-path /absolute/path/to/vosk-model-small-en-us-0.15
```

### Custom activation terms

```bash
python3 local_voice_assistant.py \
  --model-path /absolute/path/to/model \
  --activation-term "listen bot" \
  --activation-term "hey assistant"
```

### Optional device selection

List devices:

```bash
python3 local_voice_assistant.py --model-path /absolute/path/to/model --list-devices
```

Use a specific input device index:

```bash
python3 local_voice_assistant.py --model-path /absolute/path/to/model --device 1
```
