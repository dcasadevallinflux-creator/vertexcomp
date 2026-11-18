"""Downloadable models and the presets that tie node knobs to them.

Every model downloads from Hugging Face into <models dir>/<model id>/ keeping
the repository layout, so the worker can open it with diffusers offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SDXL_PIPELINE_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


@dataclass
class ModelSpec:
    id: str
    label: str
    repo: str
    files: list[str]
    size_mb: int
    kind: str              # "pipeline" | "vae" | "lora"
    license: str = ""
    note: str = ""
    requires: list[str] = field(default_factory=list)

    @property
    def main_file(self) -> str:
        return self.files[0]


MODELS: dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec("sdxl-base", "SDXL base 1.0 (fp16 weights)", "stabilityai/stable-diffusion-xl-base-1.0",
              SDXL_PIPELINE_FILES, 6940, "pipeline", "CreativeML Open RAIL++-M",
              "Base model for SDXL Generation, SDXL Lightning and LCM Turbo; also provides the config for "
              "custom checkpoints.", ["sdxl-vae-fp16-fix"]),
    ModelSpec("sdxl-turbo", "SDXL-Turbo (fp16 weights)", "stabilityai/sdxl-turbo", SDXL_PIPELINE_FILES, 6940,
              "pipeline", "Stability AI Community License (check terms for commercial use)",
              "1-4 step distilled checkpoint for the SDXL Turbo node, best at 512x512.", ["sdxl-vae-fp16-fix"]),
    ModelSpec("sdxl-config", "SDXL configs + tokenizers (for custom checkpoints)",
              "stabilityai/stable-diffusion-xl-base-1.0",
              [f for f in SDXL_PIPELINE_FILES if not f.endswith(".safetensors")], 4, "config",
              "CreativeML Open RAIL++-M",
              "Small download that lets custom .safetensors checkpoints load offline when SDXL base is not installed."),
    ModelSpec("sdxl-vae-fp16-fix", "SDXL VAE fp16-fix", "madebyollin/sdxl-vae-fp16-fix",
              ["diffusion_pytorch_model.safetensors", "config.json"], 335, "vae", "MIT",
              "VAE that decodes in fp16 without NaNs/black images; halves VAE memory."),
    ModelSpec("lightning-2step", "SDXL-Lightning 2-step LoRA", "ByteDance/SDXL-Lightning",
              ["sdxl_lightning_2step_lora.safetensors"], 394, "lora", "CreativeML Open RAIL++-M"),
    ModelSpec("lightning-4step", "SDXL-Lightning 4-step LoRA", "ByteDance/SDXL-Lightning",
              ["sdxl_lightning_4step_lora.safetensors"], 394, "lora", "CreativeML Open RAIL++-M"),
    ModelSpec("lightning-8step", "SDXL-Lightning 8-step LoRA", "ByteDance/SDXL-Lightning",
              ["sdxl_lightning_8step_lora.safetensors"], 394, "lora", "CreativeML Open RAIL++-M"),
    ModelSpec("hyper-sd-8step", "Hyper-SDXL 8-step LoRA", "ByteDance/Hyper-SD",
              ["Hyper-SDXL-8steps-lora.safetensors"], 787, "lora", "CreativeML Open RAIL++-M"),
    ModelSpec("dmd2-4step", "DMD2 SDXL 4-step LoRA (fp16)", "tianweiy/DMD2",
              ["dmd2_sdxl_4step_lora_fp16.safetensors"], 394, "lora", "CC BY-NC 4.0 (non-commercial)"),
    ModelSpec("lcm-lora-sdxl", "LCM-LoRA SDXL", "latent-consistency/lcm-lora-sdxl",
              ["pytorch_lora_weights.safetensors"], 394, "lora", "CreativeML Open RAIL++-M"),
]}


@dataclass
class Preset:
    """Recommended sampling for a speed option. `spacing` None keeps the
    scheduler's own default."""
    lora: str | None
    sampler: str
    spacing: str | None
    steps: int
    cfg: float


# Values of the `accel` knob (per node type) -> preset.
ACCEL: dict[str, Preset] = {
    "none (full steps)": Preset(None, "dpm++_2m_karras", None, 28, 6.0),
    "Lightning 2-step": Preset("lightning-2step", "euler", "trailing", 2, 1.0),
    "Lightning 4-step": Preset("lightning-4step", "euler", "trailing", 4, 1.0),
    "Lightning 8-step": Preset("lightning-8step", "euler", "trailing", 8, 1.0),
    "2-step": Preset("lightning-2step", "euler", "trailing", 2, 1.0),
    "4-step": Preset("lightning-4step", "euler", "trailing", 4, 1.0),
    "8-step": Preset("lightning-8step", "euler", "trailing", 8, 1.0),
    "Hyper-SD 8-step": Preset("hyper-sd-8step", "ddim", "trailing", 8, 1.0),
    "DMD2 4-step": Preset("dmd2-4step", "lcm", None, 4, 1.0),
    "LCM-LoRA": Preset("lcm-lora-sdxl", "lcm", None, 6, 1.5),
    "distilled checkpoint": Preset(None, "euler_a", "trailing", 2, 0.0),
}

BASE_MODELS = {"SDXL base 1.0": "sdxl-base", "SDXL-Turbo": "sdxl-turbo"}


def required_models(node_type: str, model: str, accel: str) -> list[str]:
    """Model ids a generation needs, dependencies first."""
    out: list[str] = []
    base = BASE_MODELS.get(model)
    if base is None and model == "custom checkpoint":
        base = None  # config comes from sdxl-base if present (optional)
    if base:
        out += MODELS[base].requires + [base]
    else:
        out += ["sdxl-vae-fp16-fix"]
    p = ACCEL.get(accel)
    if p and p.lora:
        out.append(p.lora)
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq
