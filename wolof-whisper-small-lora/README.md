<![CDATA[---
language:
- wo
license: mit
library_name: transformers
base_model: openai/whisper-small
tags:
- whisper
- wolof
- automatic-speech-recognition
- asr
- speech-recognition
- west-african-languages
datasets:
- alfaDF9/asr-wolof-dataset-processed-v1
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
model-index:
- name: utachicodes/dikkte-wolof-asr
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: alfaDF9/asr-wolof-dataset-processed-v1
      type: alfaDF9/asr-wolof-dataset-processed-v1
      split: test
    metrics:
    - type: wer
      value: 57.68
      name: WER
    - type: cer
      value: 37.17
      name: CER
---

[![GitHub](https://img.shields.io/badge/GitHub-dikkte__asr-black?logo=github)](https://github.com/utachicodes/dikkte_asr)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Dikkte ASR — Wolof Speech Recognition

Fine-tuned [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) for Wolof using LoRA, then merged the adapter back into the base model. This repo contains the **full merged weights** — no extra dependencies needed, just `transformers`.

Wolof is spoken by 10M+ people across Senegal, Gambia, and Mauritania but has almost no open-source ASR tooling.

## Usage

### Pipeline

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="utachicodes/dikkte-wolof-asr")
print(pipe("audio.wav")["text"])
```

### Manual

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio, torch

processor = WhisperProcessor.from_pretrained("utachicodes/dikkte-wolof-asr")
model = WhisperForConditionalGeneration.from_pretrained("utachicodes/dikkte-wolof-asr")
model.eval()

waveform, sr = torchaudio.load("audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model.generate(input_features=inputs.input_features)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

## Metrics

| Metric | Value |
|--------|-------|
| WER | 57.68% |
| CER | 37.17% |

Evaluated on 500 test samples from [`alfaDF9/asr-wolof-dataset-processed-v1`](https://huggingface.co/datasets/alfaDF9/asr-wolof-dataset-processed-v1). First pass on whisper-small — room to improve with more data or a bigger base.

## Training

| | |
|---|---|
| Base model | `openai/whisper-small` (244M params) |
| Method | LoRA — rank 32, alpha 64 |
| Targets | q_proj, v_proj, k_proj, o_proj |
| Trainable params | 5.3M (2.1%) |
| Dataset | `alfaDF9/asr-wolof-dataset-processed-v1` |
| Split | 10,380 train / 2,598 test |
| Batch size | 16 effective (2 × 8 grad accum) |
| LR | 1e-3 |
| Epochs | 3 |
| Loss | 4.21 → 0.67 |
| Precision | fp16 |
| Hardware | RTX 3060 Laptop (6GB VRAM) |
| Time | ~6 hours |

## Compared to

| Model | Params | Notes |
|-------|--------|-------|
| [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) | 1.5B | LoRA on whisper-large-v2, needs 12GB+ VRAM |
| [dofbi/wolof-asr](https://huggingface.co/dofbi/wolof-asr) | 244M | Full fine-tune, 12% WER reported |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 1B | Multilingual, Wolof adapter available |
| **dikkte** | **244M** | **LoRA-merged whisper-small, runs on 6GB VRAM** |

## Web UI

The [GitHub repo](https://github.com/utachicodes/dikkte_asr) includes a Gradio app for live mic transcription:

```bash
git clone https://github.com/utachicodes/dikkte_asr.git
cd dikkte_asr
pip install -r requirements.txt
python wolof_stt.py
```

## License

MIT

## Credits

- [alfaDF9](https://huggingface.co/alfaDF9) — Wolof ASR dataset
- [CAYTU / Seydou Diallo](https://huggingface.co/CAYTU) — whosper approach
- [OpenAI](https://github.com/openai/whisper) — Whisper
]]>