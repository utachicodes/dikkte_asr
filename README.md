<![CDATA[[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model_on_Hub-yellow)](https://huggingface.co/utachicodes/dikkte-wolof-asr)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Whisper](https://img.shields.io/badge/Base-whisper--small-orange)](https://huggingface.co/openai/whisper-small)

# Dikkte ASR

Wolof automatic speech recognition. Fine-tunes [Whisper small](https://huggingface.co/openai/whisper-small) with LoRA, merges the weights, and gives you a standard `transformers` model you can load in two lines.

Wolof is spoken by 10M+ people across Senegal, Gambia, and Mauritania — but barely has any open ASR tooling. This fixes that.

## Install

```bash
git clone https://github.com/utachicodes/dikkte_asr.git
cd dikkte_asr
pip install -r requirements.txt
```

## Usage

### From HuggingFace (easiest)

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="utachicodes/dikkte-wolof-asr")
print(pipe("audio.wav")["text"])
```

### Load the model directly

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

### Web UI (mic input)

```bash
python wolof_stt.py
# opens at http://127.0.0.1:7860
```

Record from your mic, hit transcribe. Long audio gets chunked into 30s segments automatically.

### From the raw LoRA adapter

If you want to apply the adapter yourself instead of using the merged model:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "./wolof-whisper-small-lora")
model = model.merge_and_unload()
```

## Training

| | |
|---|---|
| Base model | `openai/whisper-small` (244M params) |
| Method | LoRA — rank 32, alpha 64 |
| Targets | q_proj, v_proj, k_proj, o_proj |
| Trainable params | 5.3M (2.1% of total) |
| Dataset | [`alfaDF9/asr-wolof-dataset-processed-v1`](https://huggingface.co/datasets/alfaDF9/asr-wolof-dataset-processed-v1) |
| Split | 10,380 train / 2,598 test |
| Batch size | 16 effective (2 × 8 grad accum) |
| LR | 1e-3 |
| Epochs | 3 |
| Loss | 4.21 → 0.67 |
| Precision | fp16 |
| Hardware | RTX 3060 Laptop (6GB VRAM) |
| Time | ~6 hours |

### Retrain it yourself

```bash
python train_wolof.py
```

Downloads the dataset (~3.2GB), trains LoRA on whisper-small, saves adapter to `./wolof-whisper-small-lora/`. Edit the config block at the top to change hyperparameters.

### Evaluate

```bash
python evaluate_wer.py                    # full test set
python evaluate_wer.py --num-samples 200  # quick check
```

### Push to HuggingFace

```bash
huggingface-cli login
python push_to_hub.py                     # merged model (recommended)
python push_to_hub.py --adapter-only      # LoRA adapter only (21MB)
python push_to_hub.py --skip-eval         # skip WER computation
```

## Metrics

| Metric | Value |
|--------|-------|
| WER | 57.68% |
| CER | 37.17% |

Evaluated on 500 samples from the test split. This is a first pass on whisper-small — there's room to improve with more data, longer training, or a bigger base model.

## Compared to

| Model | Params | VRAM needed | Notes |
|-------|--------|-------------|-------|
| [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) | 1.5B | 12GB+ | LoRA on whisper-large-v2 |
| [dofbi/wolof-asr](https://huggingface.co/dofbi/wolof-asr) | 244M | ~4GB | Full fine-tune, 12% WER reported |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 1B | 8GB+ | Multilingual, Wolof adapter available |
| **dikkte** | **244M** | **~3GB** | **LoRA-merged whisper-small** |

## Files

```
train_wolof.py              Fine-tuning script (LoRA on whisper-small)
wolof_stt.py                Gradio web UI for mic transcription
evaluate_wer.py             WER/CER evaluation on test set
push_to_hub.py              Merge + push to HuggingFace Hub
test_asr.py                 Tests (pytest)
requirements.txt            Dependencies
wolof-whisper-small-lora/   LoRA adapter weights + tokenizer
```

## License

MIT

## Credits

- [alfaDF9](https://huggingface.co/alfaDF9) — Wolof ASR dataset
- [CAYTU / Seydou Diallo](https://huggingface.co/CAYTU) — whosper approach
- [OpenAI](https://github.com/openai/whisper) — Whisper
]]>