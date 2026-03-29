"""
Wolof Speech-to-Text UI
Supports: dofbi/wolof-asr, your fine-tuned LoRA model, or whosper.
"""

import os
import gradio as gr
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

CHUNK_SECONDS = 30
SAMPLE_RATE = 16000

# Pick model: use your fine-tuned LoRA if it exists, otherwise dofbi/wolof-asr
LORA_DIR = "./wolof-whisper-small-lora"
if os.path.exists(LORA_DIR):
    MODEL_ID = LORA_DIR
    USE_PEFT = True
else:
    MODEL_ID = "dofbi/wolof-asr"
    USE_PEFT = False

print(f"Loading model: {MODEL_ID} (PEFT={USE_PEFT}) ...")
processor = WhisperProcessor.from_pretrained(MODEL_ID if not USE_PEFT else "openai/whisper-small")

if USE_PEFT:
    from peft import PeftModel
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model = model.merge_and_unload()
else:
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

model.eval()

# Fix deprecated forced_decoder_ids
if model.config.forced_decoder_ids is not None:
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
    model.config.forced_decoder_ids = None

print("Model loaded!")


def transcribe(audio):
    if audio is None:
        return "No audio recorded."

    try:
        sr, data = audio

        # Convert to float32 normalized
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        # Mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample if needed
        if sr != SAMPLE_RATE:
            import torchaudio
            waveform = torch.tensor(data).unsqueeze(0)
            data = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform).squeeze().numpy()

        # Chunk long audio into 30s segments (Whisper limit)
        chunk_size = CHUNK_SECONDS * SAMPLE_RATE
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        full_text = []
        for chunk in chunks:
            inputs = processor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features=inputs.input_features,
                    attention_mask=inputs.attention_mask,
                )
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            if text.strip():
                full_text.append(text.strip())

        duration = len(data) / SAMPLE_RATE
        result = " ".join(full_text) if full_text else "(no speech detected)"
        return f"[{duration:.1f}s — {len(chunks)} chunk(s)]\n\n{result}"

    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Wolof STT") as app:
    gr.Markdown(
        f"# Wolof Speech-to-Text\n"
        f"**Model:** `{MODEL_ID}` {'(LoRA fine-tuned)' if USE_PEFT else '(pre-trained)'}"
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Click the mic to record — click again to stop",
            )
            btn = gr.Button("Transcribe", variant="primary", size="lg")
        with gr.Column():
            output = gr.Textbox(
                label="Transcription",
                lines=12,
                buttons=["copy"],
            )

    btn.click(fn=transcribe, inputs=audio_input, outputs=output)
    audio_input.stop_recording(fn=transcribe, inputs=audio_input, outputs=output)

app.launch(theme=gr.themes.Soft(), ssr_mode=False)
