"""
Tests for Dikkte Wolof ASR model.

Run with:
    pytest test_asr.py -v
    pytest test_asr.py -v -k "not slow"     # skip dataset-requiring tests
"""

import numpy as np
import pytest
import torch


LORA_DIR = "./wolof-whisper-small-lora"
BASE_MODEL = "openai/whisper-small"
SAMPLE_RATE = 16000


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def processor():
    from transformers import WhisperProcessor
    return WhisperProcessor.from_pretrained(BASE_MODEL)


@pytest.fixture(scope="session")
def merged_model():
    from transformers import WhisperForConditionalGeneration
    from peft import PeftModel
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model = model.merge_and_unload()
    # Wolof is not in Whisper-small's language list; no <|wo|> token exists.
    # Training used forced_decoder_ids=None — match that for inference.
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.eval()
    return model


@pytest.fixture(scope="session")
def silent_audio():
    """30-second silent waveform at 16 kHz."""
    return np.zeros(SAMPLE_RATE * 5, dtype=np.float32)


@pytest.fixture(scope="session")
def sine_audio():
    """5-second 440 Hz sine wave — not speech but exercises the pipeline."""
    t = np.linspace(0, 5, SAMPLE_RATE * 5, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5


# ── Adapter / config tests (no GPU required) ─────────────────────────────────

class TestAdapterFiles:
    def test_adapter_config_exists(self):
        import json, os
        cfg_path = f"{LORA_DIR}/adapter_config.json"
        assert os.path.exists(cfg_path), "adapter_config.json not found"
        with open(cfg_path) as f:
            cfg = json.load(f)
        assert cfg["peft_type"] == "LORA"
        assert cfg["r"] == 32
        assert cfg["lora_alpha"] == 64
        assert cfg["base_model_name_or_path"] == BASE_MODEL

    def test_adapter_weights_exist(self):
        import os
        weights_path = f"{LORA_DIR}/adapter_model.safetensors"
        assert os.path.exists(weights_path), "adapter_model.safetensors not found"
        size_mb = os.path.getsize(weights_path) / 1e6
        assert 15 < size_mb < 35, f"Unexpected adapter size: {size_mb:.1f} MB"

    def test_tokenizer_files_exist(self):
        import os
        required = [
            "tokenizer.json", "vocab.json", "merges.txt",
            "tokenizer_config.json", "preprocessor_config.json",
            "special_tokens_map.json",
        ]
        for fname in required:
            assert os.path.exists(f"{LORA_DIR}/{fname}"), f"Missing: {fname}"

    def test_model_card_exists(self):
        import os
        assert os.path.exists(f"{LORA_DIR}/README.md"), "Model card README.md not found"

    def test_model_card_frontmatter(self):
        with open(f"{LORA_DIR}/README.md") as f:
            content = f.read()
        assert "language:" in content
        assert "wo" in content
        assert "automatic-speech-recognition" in content or "pipeline_tag: automatic-speech-recognition" in content
        assert "license:" in content


# ── Processor tests ───────────────────────────────────────────────────────────

class TestProcessor:
    def test_processor_loads(self, processor):
        assert processor is not None

    def test_processor_feature_extractor(self, processor):
        audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        assert "input_features" in inputs
        assert inputs.input_features.shape == (1, 80, 3000)

    def test_processor_decodes_special_tokens(self, processor):
        ids = [50258, 50260, 50359, 50363]
        text = processor.decode(ids, skip_special_tokens=True)
        assert isinstance(text, str)

    def test_processor_sampling_rate(self, processor):
        assert processor.feature_extractor.sampling_rate == SAMPLE_RATE


# ── Model tests ───────────────────────────────────────────────────────────────

class TestMergedModel:
    def test_model_loads(self, merged_model):
        assert merged_model is not None

    def test_model_is_whisper(self, merged_model):
        from transformers import WhisperForConditionalGeneration
        assert isinstance(merged_model, WhisperForConditionalGeneration)

    def test_model_parameter_count(self, merged_model):
        n_params = sum(p.numel() for p in merged_model.parameters())
        assert 230_000_000 < n_params < 260_000_000, (
            f"Unexpected parameter count: {n_params:,}"
        )

    def test_forced_decoder_ids_cleared(self, merged_model):
        assert merged_model.config.forced_decoder_ids is None

    def test_generation_config_forced_decoder_ids_none(self, merged_model):
        # Model was trained with forced_decoder_ids=None (no <|wo|> token in whisper-small)
        assert merged_model.generation_config.forced_decoder_ids is None

    def _generate(self, model, processor, audio):
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            return model.generate(input_features=inputs.input_features)

    def test_forward_pass_silent_audio(self, merged_model, processor, silent_audio):
        ids = self._generate(merged_model, processor, silent_audio)
        assert ids.shape[0] == 1, "Expected batch size 1"

    def test_forward_pass_sine_audio(self, merged_model, processor, sine_audio):
        ids = self._generate(merged_model, processor, sine_audio)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        assert isinstance(text, str)

    def test_output_is_string(self, merged_model, processor, silent_audio):
        ids = self._generate(merged_model, processor, silent_audio)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        assert isinstance(text, str)

    def test_batched_inference(self, merged_model, processor):
        """Verify batch inference works (batch size 2, 30s audio = 3000 mel frames each)."""
        audio_30s = np.zeros(SAMPLE_RATE * 30, dtype=np.float32)
        inputs = processor(
            [audio_30s, audio_30s],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            ids = merged_model.generate(input_features=inputs.input_features)
        assert ids.shape[0] == 2

    def test_long_audio_chunking(self, merged_model, processor):
        """Simulate chunking a 90-second audio into 30-second chunks."""
        long_audio = np.zeros(SAMPLE_RATE * 90, dtype=np.float32)
        chunk_size = SAMPLE_RATE * 30
        chunks = [long_audio[i : i + chunk_size] for i in range(0, len(long_audio), chunk_size)]
        assert len(chunks) == 3
        results = []
        for chunk in chunks:
            ids = self._generate(merged_model, processor, chunk)
            results.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        assert len(results) == 3


# ── Pipeline API test ─────────────────────────────────────────────────────────

class TestTransformersPipeline:
    def test_pipeline_with_local_model(self, merged_model, processor):
        """Test that the model works with transformers.pipeline API."""
        try:
            from transformers import pipeline
        except Exception as exc:
            pytest.skip(f"pipeline import failed: {exc}")

        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=merged_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
            )
            audio = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
            result = pipe({"raw": audio, "sampling_rate": SAMPLE_RATE})
            assert "text" in result
            assert isinstance(result["text"], str)
        except RuntimeError as exc:
            if "torchcodec" in str(exc) or "libtorch" in str(exc):
                pytest.skip(f"torchcodec DLL not available on this platform: {exc}")
            raise


# ── Slow tests (require HuggingFace dataset download) ─────────────────────────

@pytest.mark.slow
class TestWER:
    def test_wer_below_threshold(self, merged_model, processor):
        """
        WER on 100 test samples must be below 80%.
        (Sanity check — real WER is much lower on a trained model.)
        """
        import evaluate
        from datasets import load_dataset

        wer_metric = evaluate.load("wer")
        ds = load_dataset(
            "alfaDF9/asr-wolof-dataset-processed-v1", split="test"
        ).select(range(100))

        preds, refs = [], []
        for i in range(0, len(ds), 8):
            batch = ds[i : i + 8]
            features = torch.tensor(
                np.array(batch["input_features"]), dtype=torch.float32
            )
            with torch.no_grad():
                ids = merged_model.generate(input_features=features)
            preds.extend(processor.batch_decode(ids, skip_special_tokens=True))
            refs.extend(
                processor.decode([t for t in seq if t != -100], skip_special_tokens=True)
                for seq in batch["labels"]
            )

        wer = wer_metric.compute(predictions=preds, references=refs)
        assert wer < 0.80, f"WER too high: {wer * 100:.1f}%"
