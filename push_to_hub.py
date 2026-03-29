"""
Push Dikkte Wolof ASR to Hugging Face Hub.

Merges LoRA weights into the base model and uploads:
  - Merged model weights (full WhisperForConditionalGeneration)
  - Processor / tokenizer
  - Model card with evaluation metrics

Usage:
    huggingface-cli login          # first-time auth
    python push_to_hub.py
    python push_to_hub.py --repo utachicodes/dikkte-wolof-asr
    python push_to_hub.py --adapter-only   # push LoRA adapter (smaller, 21MB)
    python push_to_hub.py --skip-eval      # skip WER eval (faster)
"""

import argparse
import textwrap
import numpy as np
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import evaluate as ev
from huggingface_hub import HfApi

LORA_DIR = "./wolof-whisper-small-lora"
BASE_MODEL = "openai/whisper-small"
DATASET_ID = "alfaDF9/asr-wolof-dataset-processed-v1"
DEFAULT_REPO = "utachicodes/dikkte-wolof-asr"
EVAL_SAMPLES = 500   # number of test samples for WER estimate


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_merged_model(device: str):
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model = model.merge_and_unload()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.eval().to(device)
    return model, processor


def compute_wer(model, processor, device: str) -> tuple[float, float]:
    """Return (WER, CER) on a fixed-size test slice."""
    wer_metric = ev.load("wer")
    cer_metric = ev.load("cer")

    ds = load_dataset(DATASET_ID, split="test")
    ds = ds.select(range(min(EVAL_SAMPLES, len(ds))))

    all_preds, all_refs = [], []
    batch_size = 8
    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        features = torch.tensor(
            np.array(batch["input_features"]), dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            ids = model.generate(input_features=features)
        preds = processor.batch_decode(ids, skip_special_tokens=True)
        refs = [
            processor.decode([t for t in seq if t != -100], skip_special_tokens=True)
            for seq in batch["labels"]
        ]
        all_preds.extend(preds)
        all_refs.extend(refs)
        print(f"  WER eval: {min(i + batch_size, len(ds))}/{len(ds)}", end="\r")

    print()
    return (
        wer_metric.compute(predictions=all_preds, references=all_refs),
        cer_metric.compute(predictions=all_preds, references=all_refs),
    )


def build_model_card(repo_id: str, wer: float | None, cer: float | None) -> str:
    wer_str = f"{wer * 100:.2f}" if wer is not None else "N/A"
    cer_str = f"{cer * 100:.2f}" if cer is not None else "N/A"

    model_index = ""
    if wer is not None:
        model_index = textwrap.dedent(f"""\
            model-index:
            - name: {repo_id}
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
                  value: {wer_str}
                  name: WER
                - type: cer
                  value: {cer_str}
                  name: CER
            """)

    card = textwrap.dedent(f"""\
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
        {model_index}---

        # Dikkte ASR — Wolof Speech Recognition

        Fine-tuned `openai/whisper-small` for Wolof (Wólof) speech recognition using
        LoRA adapters. Only 5.3M of 244M parameters were trained (2.1%), keeping the
        model fast and GPU-friendly.

        Wolof is spoken by 10M+ people across Senegal, Gambia, and Mauritania but has
        almost no open-source ASR tooling. Dikkte fills that gap.

        ## Metrics

        | Metric | Value (test split, {EVAL_SAMPLES} samples) |
        |--------|-------|
        | WER    | {wer_str}% |
        | CER    | {cer_str}% |

        ## Quick start

        ### Using `pipeline`

        ```python
        from transformers import pipeline

        pipe = pipeline(
            "automatic-speech-recognition",
            model="{repo_id}",
            generate_kwargs={{"language": "wo", "task": "transcribe"}},
        )
        result = pipe("audio.wav")
        print(result["text"])
        ```

        ### Manual usage

        ```python
        import torch, torchaudio
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained("{repo_id}")
        model = WhisperForConditionalGeneration.from_pretrained("{repo_id}")
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

        ## Training details

        | Setting | Value |
        |---------|-------|
        | Base model | `openai/whisper-small` (244M params) |
        | Method | LoRA (rank 32, alpha 64) |
        | Target modules | q\\_proj, v\\_proj, k\\_proj, o\\_proj |
        | Trainable params | 5.3M (2.1%) |
        | Dataset | `alfaDF9/asr-wolof-dataset-processed-v1` |
        | Samples | 10,380 train / 2,598 test |
        | Effective batch size | 16 (2 × 8 grad accum) |
        | Learning rate | 1e-3 |
        | Epochs | 3 |
        | Final loss | 0.67 |
        | Training regime | fp16 mixed precision |
        | GPU | RTX 3060 Laptop 6GB |
        | Training time | ~6 hours |

        ## Web UI

        A Gradio app for live mic transcription ships with the repo:

        ```bash
        git clone https://github.com/utachicodes/dikkte_asr.git
        cd dikkte_asr
        pip install -r requirements.txt
        python wolof_stt.py
        # → http://127.0.0.1:7860
        ```

        ## Comparison

        | Model | Size | Notes |
        |-------|------|-------|
        | [CAYTU/whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2) | 1.5B | LoRA on whisper-large-v2, needs 12GB+ VRAM |
        | [dofbi/wolof-asr](https://huggingface.co/dofbi/wolof-asr) | 244M | Full fine-tune, 12% WER reported |
        | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 1B | Multilingual, Wolof adapter available |
        | **{repo_id} (this model)** | **244M** | **LoRA-merged, runs on 6GB VRAM** |

        ## License

        MIT

        ## Credits

        - [alfaDF9](https://huggingface.co/alfaDF9) — Wolof ASR dataset
        - [CAYTU / Seydou Diallo](https://huggingface.co/CAYTU) — whosper LoRA approach
        - [OpenAI](https://github.com/openai/whisper) — Whisper
    """)
    return card


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help="HF Hub repo id, e.g. username/model-name")
    parser.add_argument("--adapter-only", action="store_true",
                        help="Push LoRA adapter weights only (21MB) instead of merged model")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip WER evaluation (faster push)")
    parser.add_argument("--private", action="store_true",
                        help="Create private repo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    api = HfApi()

    print(f"Target repo : {args.repo}")
    print(f"Device      : {args.device}")
    print(f"Adapter only: {args.adapter_only}")

    # ── Create repo if needed ────────────────────────────────────────────────
    try:
        api.repo_info(repo_id=args.repo, repo_type="model")
        print(f"Repo exists : {args.repo}")
    except Exception:
        print(f"Creating repo: {args.repo} …")
        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private)

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading model …")
    model, processor = load_merged_model(args.device)

    # ── WER evaluation ───────────────────────────────────────────────────────
    wer, cer = None, None
    if not args.skip_eval:
        print(f"Computing WER on {EVAL_SAMPLES} test samples …")
        wer, cer = compute_wer(model, processor, args.device)
        print(f"WER: {wer * 100:.2f}%  CER: {cer * 100:.2f}%")

    # ── Build model card ─────────────────────────────────────────────────────
    card_text = build_model_card(args.repo, wer, cer)

    # ── Push ─────────────────────────────────────────────────────────────────
    if args.adapter_only:
        print("Pushing LoRA adapter …")
        # Push adapter weights + tokenizer using PEFT's push_to_hub
        from peft import PeftModel as _PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
        peft_model = _PeftModel.from_pretrained(base, LORA_DIR)
        peft_model.push_to_hub(args.repo, private=args.private)
        processor.push_to_hub(args.repo, private=args.private)
    else:
        print("Pushing merged model …")
        model.push_to_hub(args.repo, private=args.private)
        processor.push_to_hub(args.repo, private=args.private)

    # ── Upload model card ─────────────────────────────────────────────────────
    print("Uploading model card …")
    api.upload_file(
        path_or_fileobj=card_text.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Update model card with metrics and usage",
    )

    print(f"\nDone! Model live at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
