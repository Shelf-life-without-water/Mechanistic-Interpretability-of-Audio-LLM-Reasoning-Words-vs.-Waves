
from contextlib import nullcontext
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.special import softmax
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from .config import ExperimentConfig


@dataclass
class ModelBundle:
    processor: object
    model: object
    dtype: torch.dtype
    primary_device: torch.device
    sr: int
    audio_token_id: int | None
    lm_layers: object
    num_layers: int
    hidden_size: int


def get_lm_layers(model):
    candidates = [
        "language_model.model.layers",
        "language_model.layers",
        "model.layers",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise AttributeError("Cannot locate LM layers in model.")


def load_model_bundle(config: ExperimentConfig) -> ModelBundle:
    processor = AutoProcessor.from_pretrained(
        str(config.model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(config.model_dir),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        )
    except TypeError:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(config.model_dir),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
    model.eval()
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    if hasattr(model, "padding_side"):
        model.padding_side = "right"
    primary_device = next(model.parameters()).device
    sr = processor.feature_extractor.sampling_rate
    audio_token_id = getattr(model.config, "audio_token_index", None)
    lm_layers = get_lm_layers(model)
    num_layers = len(lm_layers)
    hidden_size = model.config.text_config.hidden_size
    return ModelBundle(
        processor=processor,
        model=model,
        dtype=dtype,
        primary_device=primary_device,
        sr=sr,
        audio_token_id=audio_token_id,
        lm_layers=lm_layers,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def load_audio(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    max_abs = np.max(np.abs(wav)) if len(wav) > 0 else 1.0
    if max_abs > 1.0:
        wav = wav / max_abs
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def score_closed_set(
    bundle: ModelBundle,
    prompt_text: str,
    candidate_labels: list[str],
    audio_array: np.ndarray | None = None,
    intervention_factory=None,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
) -> dict:
    candidate_texts = [prompt_text + " " + lab for lab in candidate_labels]
    base_texts = [prompt_text for _ in candidate_labels]
    if audio_array is None:
        base_inputs = bundle.processor(text=base_texts, return_tensors="pt", padding=True)
        full_inputs = bundle.processor(text=candidate_texts, return_tensors="pt", padding=True)
    else:
        audios = [audio_array for _ in candidate_labels]
        base_inputs = bundle.processor(text=base_texts, audio=audios, return_tensors="pt", padding=True)
        full_inputs = bundle.processor(text=candidate_texts, audio=audios, return_tensors="pt", padding=True)
    prompt_lens = base_inputs["attention_mask"].sum(dim=-1).tolist()
    full_lens = full_inputs["attention_mask"].sum(dim=-1).tolist()
    full_inputs_dev = move_batch_to_device(full_inputs, bundle.primary_device)
    ctx = nullcontext()
    if intervention_factory is not None:
        ctx = intervention_factory(full_inputs, prompt_lens)
    with torch.inference_mode():
        with ctx:
            outputs = bundle.model(
                **full_inputs_dev,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
                use_cache=False,
            )
    logits = outputs.logits.detach().float().cpu()
    input_ids = full_inputs["input_ids"].cpu()
    scores = []
    tok_logps = []
    for i in range(len(candidate_labels)):
        pl = int(prompt_lens[i])
        fl = int(full_lens[i])
        tgt_ids = input_ids[i, pl:fl]
        tgt_logits = logits[i, pl - 1:fl - 1, :]
        logp = F.log_softmax(tgt_logits, dim=-1)
        token_logp = logp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        scores.append(token_logp.mean().item())
        tok_logps.append(token_logp.numpy())
    scores = np.array(scores, dtype=np.float32)
    probs = softmax(scores)
    pred = candidate_labels[int(np.argmax(scores))]
    out = {
        "scores": scores,
        "probs": probs,
        "pred": pred,
        "full_inputs_cpu": full_inputs,
        "prompt_lens": prompt_lens,
        "token_logps": tok_logps,
    }
    if output_hidden_states or output_attentions:
        out["outputs"] = outputs
    return out
