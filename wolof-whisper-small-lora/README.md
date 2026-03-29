---
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

[![GitHub](https://img.shields.io/badge/GitHub-dikkte__asr-181717?logo=github)](https://github.com/utachicodes/dikkte_asr)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/utachicodes/dikkte-wolof-asr)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.40+-FF6F00?logo=huggingface)](https://huggingface.co/docs/transformers)

# Dikkte ASR — Wolof Speech Recognition

Wolof speech-to-text model. Built by fine-tuning [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) with LoRA adapters, then merging everything into a single model. This repo has the **full merged weights** — just `pip install transformers` and go.

Wolof is spoken by 10M+ people across Senegal, Gambia, and Mauritania but barely has any open ASR tools.

## Quick start

### Install

```bash
pip install transformers torch torchaudio
```

### Transcribe audio (3 lines)

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="utachicodes/dikkte-wolof-asr")
print(pipe("audio.wav")["text"])
```

### Full control

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio, torch

processor = WhisperProcessor.from_pretrained("utachicodes/dikkte-wolof-asr")
model = WhisperForConditionalGeneration.from_pretrained("utachicodes/dikkte-wolof-asr")
model.eval()

waveform, sr = torchaudio.load("your_audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model.generate(input_features=inputs.input_features)

text = processor.batch_decode(ids, skip_special_tokens=True)[0]
print(text)
```

### GPU inference

```python
import torch
model = WhisperForConditionalGeneration.from_pretrained(
    "utachicodes/dikkte-wolof-asr",
    torch_dtype=torch.float16,
    device_map="auto",
)
```

## Web UI

Record from your mic and see the transcription live:

```bash
git clone https://github.com/utachicodes/dikkte_asr.git
cd dikkte_asr
pip install -r requirements.txt
python wolof_stt.py
# opens at http://127.0.0.1:7860
```

## Metrics

| Metric | Value |
|--------|-------|
| **WER** | 57.68% |
| **CER** | 37.17% |

Evaluated on 500 test samples. This is a v1 on whisper-small — accuracy improves with more data, longer training, or a bigger base model.

## How it was trained

| | |
|---|---|
| Base model | `openai/whisper-small` (244M params) |
| Method | LoRA (rank 32, alpha 64) |
| Targets | q_proj, v_proj, k_proj, o_proj |
| Trainable params | 5.3M (2.1% of total) |
| Dataset | [`alfaDF9/asr-wolof-dataset-processed-v1`](https://huggingface.co/datasets/alfaDF9/asr-wolof-dataset-processed-v1) |
| Samples | 10,380 train / 2,598 test |
| Effective batch | 16 (2 x 8 grad accum) |
| Learning rate | 1e-3 |
| Epochs | 3 |
| Training loss | 4.21 → 0.67 |
| Precision | fp16 mixed |
| Hardware | RTX 3060 Laptop (6GB VRAM) |
| Time | ~6 hours |

### Retrain it

```bash
git clone https://github.com/utachicodes/dikkte_asr.git
cd dikkte_asr
pip install -r requirements.txt
python train_wolof.py
```

Edit the config block at the top of `train_wolof.py` to tweak hyperparams.

## Other Wolof ASR models

| Model | Params | Notes |
|-------|--------|-------|
| [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) | 1.5B | LoRA on whisper-large-v2, needs 12GB+ VRAM |
| [dofbi/wolof-asr](https://huggingface.co/dofbi/wolof-asr) | 244M | Full fine-tune, 12% WER reported |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 1B | Multilingual, has a Wolof adapter |
| **dikkte (this)** | **244M** | **Merged LoRA on whisper-small, 6GB VRAM** |

## License

MIT

## Credits

- [alfaDF9](https://huggingface.co/alfaDF9) for the Wolof ASR dataset
- [CAYTU / Seydou Diallo](https://huggingface.co/CAYTU) for the whosper approach
- [OpenAI](https://github.com/openai/whisper) for Whisper
