# Dikkte ASR - Wolof Speech Recognition

Dikkte is an automatic speech recognition (ASR) system for **Wolof**, a language spoken by over 10 million people across Senegal, Gambia, and Mauritania. It fine-tunes OpenAI's Whisper (small) using LoRA for efficient, high-quality Wolof transcription that runs on consumer hardware.

## Approach

- **Base model:** [openai/whisper-small](https://huggingface.co/openai/whisper-small) (244M params)
- **Fine-tuning:** LoRA (Low-Rank Adaptation) via [PEFT](https://github.com/huggingface/peft) — only 5.3M trainable parameters (2.1% of the model)
- **Dataset:** [alfaDF9/asr-wolof-dataset-processed-v1](https://huggingface.co/datasets/alfaDF9/asr-wolof-dataset-processed-v1) — 10,380 training / 2,598 test samples
- **Inspired by:** [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) — same LoRA technique, but targeting a smaller model that fits on 6GB VRAM GPUs

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `openai/whisper-small` |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Batch size | 2 (x8 gradient accumulation = effective 16) |
| Learning rate | 1e-3 |
| Epochs | 3 |
| Training loss | 4.21 &rarr; 0.67 |
| Hardware | NVIDIA RTX 3060 Laptop (6GB VRAM) |
| Training time | ~6 hours |

## Quick Start

### Install

```bash
git clone https://github.com/utachicodes/dikkte_asr.git
cd dikkte_asr
pip install -r requirements.txt
```

### Run the Web UI

```bash
python wolof_stt.py
```

Opens a Gradio interface at `http://127.0.0.1:7860`:
- Click the microphone to record Wolof speech (any duration)
- Click stop — transcription runs automatically
- Long recordings are chunked into 30-second segments

### Transcribe in Python

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torchaudio
import torch

# Load model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base_model, "./wolof-whisper-small-lora")
model = model.merge_and_unload()
model.eval()

# Load audio
waveform, sr = torchaudio.load("audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

# Transcribe
inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    predicted_ids = model.generate(input_features=inputs.input_features)
text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(text)
```

## Train Your Own

To fine-tune on additional Wolof data or adjust hyperparameters:

```bash
python train_wolof.py
```

This downloads the dataset (~3.2GB), fine-tunes whisper-small with LoRA, and saves the adapter to `./wolof-whisper-small-lora`. The inference UI automatically picks it up on next run.

**Customize training** by editing the config at the top of `train_wolof.py`:

```python
BASE_MODEL = "openai/whisper-small"   # or whisper-base, whisper-medium
BATCH_SIZE = 2                         # increase if you have more VRAM
EPOCHS = 3
LR = 1e-3
LORA_R = 32
```

## Project Structure

```
dikkte_asr/
  train_wolof.py              # LoRA fine-tuning script
  wolof_stt.py                # Gradio web UI for live transcription
  requirements.txt            # Python dependencies
  wolof-whisper-small-lora/   # Fine-tuned LoRA adapter weights
    adapter_model.safetensors # LoRA weights (~21MB)
    adapter_config.json       # LoRA configuration
    tokenizer.json            # Whisper tokenizer
    ...
```

## Why Dikkte?

Wolof is a low-resource language with limited ASR tooling. Existing options:

| Model | Size | Approach |
|-------|------|----------|
| [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) | 1.5B | LoRA on whisper-large-v2 (needs 12GB+ VRAM) |
| [dofbi/wolof-asr](https://huggingface.co/dofbi/wolof-asr) | 244M | Full fine-tune of whisper-small (12% WER) |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 1B | Multilingual wav2vec2 with Wolof adapter |
| **Dikkte (this repo)** | **244M + 21MB adapter** | **LoRA on whisper-small (runs on 6GB VRAM)** |

Dikkte brings whosper-level fine-tuning to consumer GPUs.

## License

MIT

## Acknowledgments

- [alfaDF9](https://huggingface.co/alfaDF9) for the processed Wolof ASR dataset
- [CAYTU Robotics / Seydou Diallo](https://huggingface.co/CAYTU) for the whosper approach and inspiration
- [OpenAI](https://github.com/openai/whisper) for the Whisper model
- [Hugging Face](https://huggingface.co) for transformers and PEFT
