# Dikkte ASR

Wolof speech recognition. Fine-tunes Whisper (small) with LoRA so it actually works for Wolof and runs on normal hardware.

Wolof is spoken by 10M+ people across Senegal, Gambia, and Mauritania but has almost no ASR tooling. This project fixes that.

## How it works

Takes `openai/whisper-small` (244M params) and applies LoRA adapters on the attention layers. Only 5.3M parameters are trained (2.1% of the model), keeping it fast and light. Trained on [alfaDF9/asr-wolof-dataset-processed-v1](https://huggingface.co/datasets/alfaDF9/asr-wolof-dataset-processed-v1) — about 10k Wolof audio samples.

Inspired by [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) which does the same thing on whisper-large-v2 but needs 12GB+ VRAM. This runs on a laptop GPU.

## Training

| | |
|---|---|
| Base model | `openai/whisper-small` |
| LoRA rank / alpha | 32 / 64 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Effective batch size | 16 (2 x 8 grad accum) |
| Learning rate | 1e-3 |
| Epochs | 3 |
| Loss | 4.21 → 0.67 |
| Dataset | 10,380 train / 2,598 test |
| GPU | RTX 3060 Laptop 6GB |
| Time | ~6 hours |

## Setup

```bash
git clone https://github.com/utachicodes/dikkte_asr.git
cd dikkte_asr
pip install -r requirements.txt
```

## Usage

### Web UI

```bash
python wolof_stt.py
```

Opens at `http://127.0.0.1:7860`. Click mic, speak Wolof, click stop. Long audio gets chunked into 30s segments automatically.

### Python

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torchaudio, torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "./wolof-whisper-small-lora")
model = model.merge_and_unload()
model.eval()

waveform, sr = torchaudio.load("audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model.generate(input_features=inputs.input_features)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

## Retrain

```bash
python train_wolof.py
```

Downloads the dataset (~3.2GB), trains, saves adapter to `./wolof-whisper-small-lora/`. Edit the config block at the top of the script to change hyperparameters or swap the base model.

## Files

```
train_wolof.py                         Training script
wolof_stt.py                           Gradio web UI
requirements.txt                       Dependencies
wolof-whisper-small-lora/
  adapter_model.safetensors            LoRA weights (~21MB)
  adapter_config.json                  LoRA config
  tokenizer.json + vocab.json + ...    Whisper tokenizer files
```

## Comparison

| Model | Size | Notes |
|-------|------|-------|
| CAYTU/whosper-large-v2 | 1.5B | LoRA on whisper-large-v2, needs 12GB+ VRAM |
| dofbi/wolof-asr | 244M | Full fine-tune, 12% WER reported |
| facebook/mms-1b-all | 1B | Multilingual, Wolof adapter available |
| **dikkte** | **244M + 21MB** | **LoRA on whisper-small, runs on 6GB VRAM** |

## License

MIT

## Credits

- [alfaDF9](https://huggingface.co/alfaDF9) — Wolof ASR dataset
- [CAYTU / Seydou Diallo](https://huggingface.co/CAYTU) — whosper approach
- [OpenAI](https://github.com/openai/whisper) — Whisper
