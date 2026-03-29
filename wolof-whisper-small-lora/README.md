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
- lora
- peft
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

# Dikkte ASR — Wolof Speech Recognition

Fine-tuned `openai/whisper-small` for Wolof (Wólof) speech recognition using LoRA adapters.
Only 5.3M of 244M parameters were trained (2.1%), keeping it fast and laptop-GPU-friendly.

Wolof is spoken by 10M+ people across Senegal, Gambia, and Mauritania but has almost no
open-source ASR tooling. Dikkte fills that gap.

## Quick start

### Using `pipeline` (easiest)

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="utachicodes/dikkte-wolof-asr",
    generate_kwargs={"language": "wo", "task": "transcribe"},
)
result = pipe("audio.wav")
print(result["text"])
```

### Manual usage

```python
import torch, torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("utachicodes/dikkte-wolof-asr")
model = WhisperForConditionalGeneration.from_pretrained("utachicodes/dikkte-wolof-asr")
model.eval()

waveform, sr = torchaudio.load("audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

inputs = processor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt",
)
with torch.no_grad():
    ids = model.generate(
        input_features=inputs.input_features,
        language="wo",
        task="transcribe",
    )
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

### Using the raw LoRA adapter (advanced)

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torchaudio, torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "utachicodes/dikkte-wolof-asr")
model = model.merge_and_unload()
model.eval()

waveform, sr = torchaudio.load("audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model.generate(input_features=inputs.input_features, language="wo", task="transcribe")
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

## Training details

| Setting | Value |
|---------|-------|
| Base model | `openai/whisper-small` (244M params) |
| Method | LoRA (rank 32, alpha 64) |
| Target modules | q\_proj, v\_proj, k\_proj, o\_proj |
| Trainable params | 5.3M (2.1%) |
| Dataset | `alfaDF9/asr-wolof-dataset-processed-v1` |
| Samples | 10,380 train / 2,598 test |
| Effective batch size | 16 (2 × 8 grad accum) |
| Learning rate | 1e-3 |
| Epochs | 3 |
| Final loss | 4.21 → 0.67 |
| Training regime | fp16 mixed precision |
| GPU | RTX 3060 Laptop 6GB |
| Training time | ~6 hours |
| PEFT version | 0.18.1 |

## Comparison

| Model | Size | Notes |
|-------|------|-------|
| [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) | 1.5B | LoRA on whisper-large-v2, needs 12GB+ VRAM |
| [dofbi/wolof-asr](https://huggingface.co/dofbi/wolof-asr) | 244M | Full fine-tune, 12% WER reported |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 1B | Multilingual, Wolof adapter available |
| **dikkte (this model)** | **244M** | **LoRA on whisper-small, runs on 6GB VRAM** |

## Web UI

```bash
git clone https://github.com/utachicodes/dikkte_asr.git
cd dikkte_asr
pip install -r requirements.txt
python wolof_stt.py
# → http://127.0.0.1:7860
```

## License

MIT

## Credits

- [alfaDF9](https://huggingface.co/alfaDF9) — Wolof ASR dataset
- [CAYTU / Seydou Diallo](https://huggingface.co/CAYTU) — whosper LoRA approach
- [OpenAI](https://github.com/openai/whisper) — Whisper
