---
name: finetune-locally
description: Fine-tune a Hugging Face model on a local dataset using local GPU/CPU, without submitting to HF Jobs or pushing to the Hub.
---

# Fine-tune locally (no HF Jobs, no Hub upload)

Use this recipe when the user wants to fine-tune a model on their own
machine with their own data and no cloud round-trips.

## Ground rules

- Drive everything through `Bash` (locally). Never call `hf_jobs`; it's
  filtered off when the SDK backend is active, but even if available,
  don't use it.
- Load data from the local filesystem. `datasets.load_dataset("json",
  data_files="/path/to/train.jsonl")` is the canonical pattern for
  user-supplied JSONL.
- Never set `push_to_hub=True` unless the user explicitly asks. Save to
  a local directory with `trainer.save_model("/path/to/output")`.
- Prefer LoRA / PEFT over full fine-tuning for exploratory runs — a
  135M-parameter model produces ~500MB on `save_model()`, which adds up
  fast across experiments.
- Disable CUDA explicitly when the user says "CPU only" or when a
  `torch.cuda.is_available()` pre-check fails.

## Pre-flight checks

Before writing the training script, one Bash call:

```bash
python - <<'PY'
import torch
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device", torch.cuda.get_device_name(0))
print("torch", torch.__version__)
PY
```

If CUDA is reported unavailable despite an NVIDIA GPU being visible to
`nvidia-smi`, it's almost certainly a driver/runtime mismatch (Spike 5
hit this on driver 595.58.03 vs torch cu121/cu128/cu130). Tell the user
and fall back to CPU rather than chasing it.

## Minimal LoRA recipe

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

ds = load_dataset("json", data_files="/path/to/train.jsonl", split="train")
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules=["q_proj", "v_proj"])

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    peft_config=peft_cfg,
    args=SFTConfig(
        output_dir="/path/to/output",
        max_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=False, fp16=False,  # toggle based on hardware
        report_to="none",        # no trackio unless user asks
        push_to_hub=False,
    ),
)
trainer.train()
trainer.save_model("/path/to/output")
```

## Monitoring

If the training is long (>2 min), emit a brief status summary after the
run: final loss, steps, wall time, output dir contents. Don't stream
every step — noise.

## When to refuse

If the user asks to "fine-tune on the Hub" or provides an `org/repo`
path and no local dataset, clarify: "This recipe is local-first. If
you'd like to use HF Jobs or read from the Hub, re-run with
`--enable-hf-infra`." Don't silently upload.
