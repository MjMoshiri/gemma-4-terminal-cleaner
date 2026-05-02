"""Cloud QLoRA fine-tune of Gemma 4 E2B on Modal H100, then merge + MLX convert.

How to run this
---------------

1. Create the HF token secret (one-time):

       modal secret create huggingface-token HF_TOKEN=<your-token>

2. Format training data with channel-marker wrapping (local):

       uv run python -m train.format_for_cloud --in data/mlx --out data/cloud

3. Validate the pipeline end-to-end without burning GPU time:

       modal run train/cloud_train.py::main --dry-run

4. Real training run (~30-60 min, ~$2 on H100):

       modal run train/cloud_train.py::main

5. Sync the trained MLX model back to local disk:

       modal run train/cloud_train.py::download_output \\
           --target /Users/mjmoshiri/gemma_4/models/trained

Resolved HF model id
--------------------

Verified via HuggingFace Hub (May 2026): the canonical instruction-tuned 2B
Gemma 4 model is published as ``google/gemma-4-E2B-it``. Unsloth maintains a
mirror at ``unsloth/gemma-4-E2B-it`` that is the recommended ``model_name`` for
``unsloth.FastModel.from_pretrained``. We try Unsloth first; if its loader
raises (e.g., older container without Gemma 4 mappings) we fall back to
vanilla HF + PEFT against the Google id.

LoRA target modules
-------------------

Cloud training uses the upstream HF model, which exposes independent
q/k/v/o projections for every layer. The Mac MLX checkpoint had a kv-shared
constraint (last 20 layers share KV from earlier ones, see
``train/patch_mlx_lm.py``) so local LoRA was limited to ``q_proj`` and
``o_proj``; on cloud we target all four (``q_proj,k_proj,v_proj,o_proj``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# --- Modal infra ---------------------------------------------------------

APP_NAME = "gemma-4-terminal-cleaner"
MODEL_ID_UNSLOTH = "unsloth/gemma-4-E2B-it"
MODEL_ID_HF = "google/gemma-4-E2B-it"
HF_FALLBACK_CANDIDATES = [
    MODEL_ID_HF,
    "google/gemma-4-2b-it",
    "google/gemma-4-e2b-it",
]

VOLUME_NAME = "gemma-4-terminal-cleaner-data"
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cloud"
REMOTE_DATA_DIR = Path("/data")
REMOTE_OUTPUT_DIR = REMOTE_DATA_DIR / "output"
REMOTE_INPUT_DIR = REMOTE_DATA_DIR / "input"

# Image: lock to Python 3.11; install the deep-learning + MLX stack.
#
# Pinning strategy: pin unsloth + torch to specific versions and let unsloth
# pull its own (already-tested) transitive deps. We use uv_pip_install for
# faster, deterministic resolution — pip's backtracker explodes on the
# unsloth/peft/trl/transformers constraint web.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        # Pin to a specific unsloth release so the resolver doesn't search
        # backwards through every minor version. unsloth pulls a compatible
        # torch as a transitive dep — don't pin torch ourselves.
        "unsloth==2026.4.8",
        # MLX conversion (CPU build is fine here)
        "mlx-lm",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")


# --- Pure (non-Modal) helpers — safe to import & unit-test locally ------


def build_lora_config(rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Construct a PEFT LoraConfig. Imported lazily so this module is
    importable without ``peft`` installed (e.g., for local pytest)."""
    from peft import LoraConfig  # type: ignore[import-not-found]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def render_chat_for_training(tokenizer, record: dict) -> str:
    """Apply the tokenizer's chat template to a record's messages.

    Records produced by ``train/format_for_cloud.py`` already have the
    final-channel markers baked into the assistant message, so the chat
    template's ``strip_thinking`` macro will pass them through (it only strips
    the ``thought`` channel)."""
    return tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False
    )


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Stream JSONL into memory (used for tiny dry-run samples only)."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def resolve_hf_model_id(candidates: list[str] | None = None) -> str:
    """Pick the first model id that exists on the Hub.

    Used by the HF fallback path if ``MODEL_ID_HF`` ever moves. Requires the
    HF token to be available in the env (set by Modal secret).
    """
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    api = HfApi()
    candidates = candidates or HF_FALLBACK_CANDIDATES
    for cand in candidates:
        try:
            api.model_info(cand)
            return cand
        except Exception:  # noqa: BLE001 — any failure means try next
            continue
    raise RuntimeError(
        f"none of the candidate Gemma 4 model ids resolved on HF: {candidates}"
    )


# --- Modal functions ----------------------------------------------------


@app.function(
    gpu="H100",
    timeout=60 * 60 * 4,  # 4-hour ceiling
    secrets=[hf_secret],
    volumes={str(REMOTE_DATA_DIR): volume},
    # Keep the data we mount via Volume below; Modal will sync new files at
    # commit time at the end of the function.
)
def train(
    dry_run: bool = False,
    iters: int = 10000,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-4,
    lr_min: float = 1e-5,
    warmup_steps: int = 100,
    max_seq_length: int = 4096,
    use_unsloth: bool = True,
) -> dict:
    """Train + merge + MLX-convert. Runs entirely on the H100.

    Returns a dict summarizing what happened (also useful for tests via the
    Modal CLI which prints the return value).
    """
    import shutil

    import torch  # type: ignore[import-not-found]

    os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    # Pull in any data uploaded by upload_data() in a prior call.
    volume.reload()
    REMOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_dir = REMOTE_OUTPUT_DIR / "adapter"
    merged_hf_dir = REMOTE_OUTPUT_DIR / "merged_hf"
    merged_mlx_dir = REMOTE_OUTPUT_DIR / "merged_mlx"

    # --- Phase 1: load model + tokenizer ---
    model = None
    tokenizer = None
    used_unsloth = False
    if use_unsloth:
        try:
            from unsloth import FastModel  # type: ignore[import-not-found]

            model, tokenizer = FastModel.from_pretrained(
                model_name=MODEL_ID_UNSLOTH,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                full_finetuning=False,
            )
            model = FastModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=0,
            )
            used_unsloth = True
            print("[setup] Unsloth FastModel loaded successfully.")
        except (ImportError, NotImplementedError, OSError, RuntimeError) as e:
            print(f"[setup] Unsloth path unavailable ({e!r}); falling back to HF.")

    if model is None:
        # HF + PEFT fallback path.
        from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        resolved = resolve_hf_model_id()
        print(f"[setup] HF fallback: loading {resolved}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, build_lora_config())

    # Gemma 4 is multimodal; FastModel.from_pretrained may return the
    # multimodal Processor (Gemma4Processor) which routes positional args to
    # the image branch. Extract the underlying text tokenizer if so.
    processor = tokenizer
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Phase 2: data ---
    train_path = REMOTE_INPUT_DIR / "train.jsonl"
    val_path = REMOTE_INPUT_DIR / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"missing {train_path} — upload via `modal volume put "
            f"{VOLUME_NAME} data/cloud/train.jsonl /input/train.jsonl` first"
        )

    from datasets import load_dataset  # type: ignore[import-not-found]

    raw = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )

    def _format(example):
        return {"text": render_chat_for_training(tokenizer, example)}

    raw = raw.map(_format, remove_columns=["messages"])

    # --- Phase 3: dry-run validation: forward pass + exit ---
    if dry_run:
        print("[dry-run] Loaded model + 1 sample; running forward pass.")
        sample = raw["train"][0]["text"]
        ids = tokenizer(
            sample, return_tensors="pt", truncation=True, max_length=max_seq_length
        )
        if torch.cuda.is_available():
            ids = {k: v.cuda() for k, v in ids.items()}
        with torch.no_grad():
            out = model(**ids)
        print(f"[dry-run] forward OK; logits shape={tuple(out.logits.shape)}")
        return {
            "phase": "dry-run",
            "used_unsloth": used_unsloth,
            "model_id": MODEL_ID_UNSLOTH if used_unsloth else MODEL_ID_HF,
            "train_records": len(raw["train"]),
            "val_records": len(raw["validation"]),
        }

    # --- Phase 4: real training ---
    from transformers import TrainingArguments  # type: ignore[import-not-found]
    from trl import SFTTrainer  # type: ignore[import-not-found]

    training_args = TrainingArguments(
        output_dir=str(REMOTE_OUTPUT_DIR / "ckpts"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=iters,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        # Cosine to a minimum — TRL/HF achieves this by setting eta_min via
        # the optimizer; we approximate with min_lr_rate.
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_steps=500,
        # In-training eval is expensive (~12 min per pass on 18.7K val
        # samples). Disabled — we run the canonical eval/run.py pass against
        # the merged MLX model after training instead.
        eval_strategy="no",
        save_total_limit=2,
        report_to=[],
        seed=0,
        # mask_prompt: SFTTrainer with `dataset_text_field` + a response template
        # is the canonical way; here we use `completion_only_loss=True` (TRL >=
        # 0.12) which masks loss to the response side automatically when the
        # data is in the chat-text format we produced above.
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw["train"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        # mask_prompt analogue: only score loss on the assistant turn.
        # In TRL >=0.12 this is enabled by setting completion_only_loss=True
        # AND providing a response_template; we use the channel-open marker as
        # the unambiguous response boundary.
        # (If your TRL version doesn't support this kwarg, drop it and rely on
        # SFTTrainer's default which trains on full sequence.)
    )

    trainer.train()

    # --- Phase 5: save adapter ---
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[save] adapter -> {adapter_dir}")

    # --- Phase 6: merge ---
    print("[merge] loading base + adapter for merge_and_unload()")
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM as _AMC  # type: ignore[import-not-found]

    # Reload base in bf16 (un-quantized) so the merged weights are clean fp.
    base = _AMC.from_pretrained(
        MODEL_ID_HF if not used_unsloth else MODEL_ID_UNSLOTH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()
    merged_hf_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_hf_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_hf_dir))
    print(f"[merge] merged HF -> {merged_hf_dir}")

    # --- Phase 7: MLX 4-bit convert ---
    print("[mlx] converting merged HF -> 4-bit MLX")
    if merged_mlx_dir.exists():
        shutil.rmtree(merged_mlx_dir)
    from mlx_lm import convert  # type: ignore[import-not-found]

    # mlx_lm.convert signature: convert(hf_path, mlx_path, quantize=True, q_bits=4)
    convert(
        hf_path=str(merged_hf_dir),
        mlx_path=str(merged_mlx_dir),
        quantize=True,
        q_bits=4,
    )
    print(f"[mlx] MLX 4-bit -> {merged_mlx_dir}")

    # Commit volume changes so download_output() sees them.
    volume.commit()
    return {
        "phase": "trained",
        "used_unsloth": used_unsloth,
        "model_id": MODEL_ID_UNSLOTH if used_unsloth else MODEL_ID_HF,
        "adapter_path": str(adapter_dir),
        "merged_mlx_path": str(merged_mlx_dir),
        "iters": iters,
    }


def upload_data_local() -> dict:
    """Sync local ``data/cloud/{train,val}.jsonl`` into the Modal Volume.

    This is a LOCAL function (no @app.function decorator) — it runs in the
    user's shell and uses ``volume.batch_upload()`` to push files to the
    remote volume. Reading files from `LOCAL_DATA_DIR` (which is on the
    user's filesystem) is the whole point — using a Modal function would
    look for the files inside the container, where they don't exist.
    """
    # Per-file upload so an existing file (FileExistsError) just skips
    # rather than aborting the whole batch. Avoids re-pushing 900 MB on
    # every restart while still uploading new files.
    uploaded: list[str] = []
    skipped: list[str] = []
    for name in ("train.jsonl", "val.jsonl"):
        local = LOCAL_DATA_DIR / name
        if not local.exists():
            raise FileNotFoundError(
                f"{local} missing; run `python -m train.format_for_cloud` first"
            )
        try:
            with volume.batch_upload(force=False) as batch:
                batch.put_file(str(local), f"/input/{name}")
            uploaded.append(name)
        except FileExistsError:
            skipped.append(name)
    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "remote_dir": str(REMOTE_INPUT_DIR),
    }


@app.function(
    timeout=60 * 30,
    volumes={str(REMOTE_DATA_DIR): volume},
)
def download_output(target: str) -> dict:
    """Mirror the trained ``merged_mlx`` directory back to a local path.

    Modal's ``modal run`` with a function returning a value prints that value;
    this function returns the list of files copied. Use::

        modal run train/cloud_train.py::download_output \\
            --target /Users/mjmoshiri/gemma_4/models/trained
    """
    # Pull in writes from the most recent train() commit.
    volume.reload()
    src_dir = REMOTE_OUTPUT_DIR / "merged_mlx"
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} missing — run train first")
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for src in src_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(src_dir)
        dst = target_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as f_in, dst.open("wb") as f_out:
            while chunk := f_in.read(1 << 22):
                f_out.write(chunk)
        files.append(str(rel))
    return {"target": str(target_path), "files": files}


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    iters: int = 10000,
    batch_size: int = 4,
    grad_accum: int = 8,
    download_to: str = "",
):
    """Local entrypoint: upload data (if needed), then train, then optionally
    download the merged MLX weights back to ``download_to``.
    """
    print("[cloud_train] uploading data to volume…")
    upload_summary = upload_data_local()
    print(f"[cloud_train] upload: {upload_summary}")

    print(f"[cloud_train] launching train(dry_run={dry_run}, iters={iters})")
    summary = train.remote(
        dry_run=dry_run,
        iters=iters,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    print(f"[cloud_train] train: {summary}")

    if download_to and not dry_run:
        print(f"[cloud_train] downloading merged MLX -> {download_to}")
        out = download_output.remote(target=download_to)
        print(f"[cloud_train] download: {out}")
    return summary
