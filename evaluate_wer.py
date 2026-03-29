"""
Evaluate Word Error Rate (WER) of the Dikkte Wolof ASR model on the test split.

Usage:
    python evaluate_wer.py
    python evaluate_wer.py --num-samples 200   # quick smoke-test
    python evaluate_wer.py --device cpu
"""

import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import evaluate

LORA_DIR = "./wolof-whisper-small-lora"
BASE_MODEL = "openai/whisper-small"
DATASET_ID = "alfaDF9/asr-wolof-dataset-processed-v1"
SAMPLE_RATE = 16000
BATCH_SIZE = 8


def load_model(device: str):
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model = model.merge_and_unload()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.eval().to(device)
    return model, processor


def run_eval(model, processor, dataset, device: str, batch_size: int):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    all_preds = []
    all_refs = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        input_features = torch.tensor(
            np.array(batch["input_features"]), dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features=input_features)

        preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # Decode reference labels (strip padding token -100)
        label_ids = batch["labels"]
        refs = []
        for seq in label_ids:
            filtered = [t for t in seq if t != -100]
            refs.append(processor.decode(filtered, skip_special_tokens=True))

        all_preds.extend(preds)
        all_refs.extend(refs)

        if (i // batch_size) % 5 == 0:
            done = min(i + batch_size, len(dataset))
            print(f"  {done}/{len(dataset)} samples processed …")

    wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    cer = cer_metric.compute(predictions=all_preds, references=all_refs)
    return wer, cer, all_preds, all_refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Evaluate on first N test samples (default: all)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--show-examples", type=int, default=5,
                        help="Print N prediction/reference pairs")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print("Loading model …")
    model, processor = load_model(args.device)

    print(f"Loading dataset: {DATASET_ID} …")
    ds = load_dataset(DATASET_ID, split="test")
    if args.num_samples:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"Evaluating on {len(ds)} samples …")

    wer, cer, preds, refs = run_eval(
        model, processor, ds, args.device, args.batch_size
    )

    print("\n" + "=" * 50)
    print(f"  WER : {wer * 100:.2f}%")
    print(f"  CER : {cer * 100:.2f}%")
    print("=" * 50)

    if args.show_examples:
        print(f"\nSample predictions (first {args.show_examples}):")
        for ref, pred in zip(refs[: args.show_examples], preds[: args.show_examples]):
            print(f"  REF : {ref}")
            print(f"  PRED: {pred}")
            print()


if __name__ == "__main__":
    main()
