#!/usr/bin/env python3
"""
Stage 1: Format Learning — LoRA fine-tune Bonsai 1-bit 8B

Teaches Bonsai to:
  1. Emit <think> traces with goal/constraints/plan/checks/decision schema
  2. Stay within token budgets (40-500 by difficulty)
  3. Produce valid structured output
  4. Reason before answering

Uses Unsloth for efficient LoRA training on consumer GPU (RTX 5070 Ti).
"""

from pathlib import Path
import torch
try:
    from torch.utils import _pytree
    if not hasattr(_pytree, "register_constant"):
        def _register_constant(*args, **kwargs):
            return None
        _pytree.register_constant = _register_constant
except Exception:
    pass
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# Config — from Gemini.txt spec with adjustments
# ---------------------------------------------------------------------------
BASE_DIR = str(Path(__file__).parent)
TRAIN_FILE = f"{BASE_DIR}/karpathy_loop_output/train_rebalanced.jsonl"
EVAL_FILE = f"{BASE_DIR}/karpathy_loop_output/eval_rebalanced.jsonl"
OUTPUT_DIR = f"{BASE_DIR}/karpathy_loop_output/stage3_rebalanced_checkpoint"

# Model — Bonsai is Qwen3-8B based (confirmed from GGUF metadata)
# GGUF default path: ~/.lmstudio/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B.gguf
MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"  # Unsloth's optimized Qwen3-8B
MAX_SEQ_LENGTH = 2048   # Fits in 16GB VRAM with QLoRA. Our traces avg 101 tokens.
DTYPE = None             # Auto-detect (float16 on Ampere+)
LOAD_IN_4BIT = True      # QLoRA

# LoRA — from Gemini.txt
LORA_R = 64              # Higher rank compensates for low-bit quantization
LORA_ALPHA = 64           # Alpha = rank for stable training
LORA_DROPOUT = 0.05
TARGET_MODULES = [         # All linear layers per spec
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
# Post-training: TurboQuant KV cache compression for inference
# V=4-bit, K=3-bit — values are more sensitive than keys
# Enables 32K-64K context without VRAM spike or coherence drift
# Install: pip install turboquant
# Usage: from turboquant import TurboQuantCache
#        cache = TurboQuantCache(model, v_bits=4, k_bits=3)
#        model.generate(..., past_key_values=cache, use_cache=True)

LEARNING_RATE = 1.5e-4    # From spec — prevents "Bit-Collapse" in 1-bit weights
BATCH_SIZE = 1            # Minimal for 16GB VRAM
GRADIENT_ACCUMULATION = 16 # Effective batch = 16
NUM_EPOCHS = 3            # Format learning — 3 epochs is usually enough
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100

# ---------------------------------------------------------------------------
# ChatML template for Qwen models
# ---------------------------------------------------------------------------
CHATML_TEMPLATE = """{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}
<|im_end|>
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def main():
    print("=" * 60)
    print("Stage 1: Format Learning — Bonsai LoRA Fine-Tune")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # -----------------------------------------------------------------------
    # 2. Add LoRA adapters
    # -----------------------------------------------------------------------
    print(f"Adding LoRA: rank={LORA_R}, targets={TARGET_MODULES}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=42,
    )

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # -----------------------------------------------------------------------
    # 3. Load dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset: {TRAIN_FILE}")
    dataset = load_dataset("json", data_files={
        "train": TRAIN_FILE,
        "eval": EVAL_FILE,
    })
    print(f"Train rows: {len(dataset['train'])}, Eval rows: {len(dataset['eval'])}")

    # -----------------------------------------------------------------------
    # 4. Keep only ChatML conversation rows and format them for training
    # -----------------------------------------------------------------------
    def only_chatml(example):
        return "conversations" in example and example["conversations"] is not None

    def format_chatml(example):
        """Convert conversations list to ChatML string."""
        messages = example["conversations"]
        text = ""
        for msg in messages:
            text += f"<|im_start|>{msg['role']}\n{msg['content']}\n<|im_end|>\n"
        return {"text": text}

    dataset = dataset.filter(only_chatml)
    print(f"ChatML rows: Train {len(dataset['train'])}, Eval {len(dataset['eval'])}")
    dataset = dataset.map(format_chatml)

    # -----------------------------------------------------------------------
    # 5. Train
    # -----------------------------------------------------------------------
    print(f"\nStarting training:")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Output: {OUTPUT_DIR}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            eval_strategy="no",
            save_strategy="no",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            report_to="none",  # Set to "wandb" if WANDB_API_KEY is configured
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            packing=True,  # Pack short examples together for efficiency
        ),
    )

    # Check GPU memory before training
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")

    print("\nTraining...")
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Train loss: {stats.training_loss:.4f}")
    print(f"  Steps: {stats.global_step}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

    # -----------------------------------------------------------------------
    # 6. Save
    # -----------------------------------------------------------------------
    print(f"\nSaving to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Deliverable for this run: adapter checkpoint only.
    # Manual GGUF conversion happens outside the trainer to avoid the long finalize/export tail.
    print(f"\n{'=' * 60}")
    print("Stage 1 complete. Adapter checkpoint saved.")
    print(f"  Checkpoint dir: {OUTPUT_DIR}")
    print("  Manual GGUF conversion will be done separately.")
    print(f"{'=' * 60}")
    return


if __name__ == "__main__":
    main()
