"""
Fine-tune whisper-small on Wolof ASR data using LoRA (PEFT).
Same technique as whosper-large-v2, but on a smaller base model.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
BASE_MODEL = "openai/whisper-small"
DATASET_ID = "alfaDF9/asr-wolof-dataset-processed-v1"
OUTPUT_DIR = "./wolof-whisper-small-lora"
BATCH_SIZE = 2        # safe for 6GB VRAM (RTX 3060)
GRAD_ACCUM = 8        # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS = 3
LR = 1e-3
LORA_R = 32
LORA_ALPHA = 64
MAX_LABEL_LEN = 448   # whisper max token length


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model & processor ──────────────────────────────────────
    print(f"Loading {BASE_MODEL} ...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    # Clear forced decoder IDs so the model learns freely
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # ── Apply LoRA ──────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # ── Load dataset ────────────────────────────────────────────────
    print(f"Loading dataset: {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)
    print(f"Train: {len(ds['train'])} samples, Test: {len(ds['test'])} samples")

    # The dataset has pre-extracted input_features and tokenized labels
    def collate_fn(batch):
        input_features = torch.tensor(
            np.array([item["input_features"] for item in batch]),
            dtype=torch.float32,
        )

        labels = [item["labels"] for item in batch]
        # Pad labels to same length, use -100 for padding (ignored by loss)
        max_len = min(max(len(l) for l in labels), MAX_LABEL_LEN)
        padded_labels = []
        for l in labels:
            l = l[:MAX_LABEL_LEN]
            padded = l + [-100] * (max_len - len(l))
            padded_labels.append(padded)

        return {
            "input_features": input_features,
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    # ── Training args ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=100,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate_fn,
    )

    # ── Train ───────────────────────────────────────────────────────
    print("Starting training ...")
    trainer.train()

    # ── Save ────────────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nDone! Model saved to {OUTPUT_DIR}")
    print("Use it with:")
    print(f'  python wolof_stt.py  (after updating MODEL_ID to "{OUTPUT_DIR}")')


if __name__ == "__main__":
    main()
