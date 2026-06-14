import os
import glob
import random
import argparse
import warnings

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
    MultipleNegativesSymmetricRankingLoss,
)

# Reuse the corpus-level boilerplate stripping that embedding.py uses, so the
# paired text side is cleaned the same way the search index is.
from embedding import find_boilerplate, document_lines

# Domain-adapt this base model on the vehicle listings in text/. The result is
# a custom model saved to MODEL_DIR, which embedding.py and search.py then load.
# mpnet-base (768-dim, 12 layers) trades speed for quality over MiniLM-L6.
BASE_MODEL = "all-mpnet-base-v2"
TEXT_DIR = "text"
DESC_DIR = "descriptions"
MODEL_DIR = "model"
# VINs held out of paired training, written for evaluate.py to score against.
HOLDOUT_FILE = os.path.join(MODEL_DIR, "holdout.txt")

# Lines shorter than this are dropped — they're mostly nav fragments ("NEW",
# "Trucks") that carry no vehicle-specific meaning.
MIN_CHARS = 20


def load_chunks(text_dir=TEXT_DIR):
    """Read text/*.txt and return the unique, content-bearing lines.

    The listings are line-oriented. We split on lines, drop short fragments,
    and globally de-duplicate so the dealer navigation boilerplate (identical
    across all 411 pages) doesn't drown out the vehicle-specific text.
    """
    seen, chunks = set(), []
    for path in sorted(glob.glob(os.path.join(text_dir, "*.txt"))):
        with open(path, encoding="utf-8") as text_file:
            for line in text_file:
                line = line.strip()
                if len(line) < MIN_CHARS or line in seen:
                    continue
                seen.add(line)
                chunks.append(line)
    return chunks


def load_pairs(text_dir=TEXT_DIR, desc_dir=DESC_DIR):
    """Return [(vin, cleaned_text, description), ...] for VINs that have both.

    The text side is boilerplate-stripped (shared dealer chrome removed), the
    same cleaning embedding.py applies, so training and indexing see the same
    text. VINs missing either file are skipped.
    """
    candidates = []
    for desc_path in sorted(glob.glob(os.path.join(desc_dir, "*.txt"))):
        vin = os.path.splitext(os.path.basename(desc_path))[0]
        text_path = os.path.join(text_dir, "%s.txt" % vin)
        if os.path.exists(text_path):
            candidates.append((vin, text_path, desc_path))
    if not candidates:
        return []

    boilerplate = find_boilerplate([t for _, t, _ in candidates])
    pairs = []
    for vin, text_path, desc_path in candidates:
        lines = [l for l in document_lines(text_path) if l not in boilerplate]
        text = "\n".join(lines)
        with open(desc_path, encoding="utf-8") as desc_file:
            description = desc_file.read().strip()
        if text and description:
            pairs.append((vin, text, description))
    return pairs


def expand_pairs(pairs, model):
    """Turn (vin, text, description) into (chunk, description) training examples.

    Each listing is split into token-window chunks that fit one forward pass, and
    every chunk is paired with the same description — augmenting ~400 listings
    into several thousand pairs and avoiding the max_seq_length truncation that a
    single (whole-text, description) pair would suffer.
    """
    window = model.max_seq_length - 2  # leave room for [CLS]/[SEP]
    tokenizer = model.tokenizer
    examples = []
    for _vin, text, description in pairs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(ids), window):
            chunk = tokenizer.decode(ids[start:start + window])
            if chunk.strip():
                examples.append((chunk, description))
    return examples


def write_holdout(vins):
    """Record the held-out VINs (one per line) for evaluate.py; clears stale."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HOLDOUT_FILE, "w", encoding="utf-8") as holdout_file:
        holdout_file.write("\n".join(sorted(vins)))


def to_device(features, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }


def ensure_model_writable():
    """Fail fast if MODEL_DIR is open in another process.

    On Windows, a process that has the model loaded (e.g. the webapp) holds
    model.safetensors memory-mapped, so save() fails with a cryptic OS error
    1224 — but only at the first checkpoint, after training has already run.
    Probe for the lock up front by renaming the file and back.
    """
    target = os.path.join(MODEL_DIR, "model.safetensors")
    if not os.path.exists(target):
        return  # fresh save — nothing to conflict with
    probe = target + ".lockcheck"
    try:
        os.replace(target, probe)
    except OSError:
        raise SystemExit(
            "%s is open in another process — likely the webapp (`python app.py`).\n"
            "Stop it before retraining, then re-run finetune.py." % target
        )
    os.replace(probe, target)  # free — restore and proceed


def build_examples(model, paired, holdout, seed):
    """Build the list of (text_a, text_b) training pairs for the chosen mode.

    paired: (text chunk, its description) — real positive pairs across two views.
    SimCSE: (chunk, chunk) — the same text twice; dropout makes the views differ.
    """
    if paired:
        pairs = load_pairs()
        if not pairs:
            print("--paired requested but no text+description pairs found; "
                  "falling back to unsupervised SimCSE")
            paired = False

    if paired:
        random.Random(seed).shuffle(pairs)
        held = set()
        if holdout > 0:
            n_hold = max(1, int(len(pairs) * holdout))
            held = {vin for vin, _, _ in pairs[:n_hold]}
        write_holdout(held)
        train_pairs = [p for p in pairs if p[0] not in held]
        if held:
            print("holdout: %d VINs reserved for evaluate.py; training on %d"
                  % (len(held), len(train_pairs)))
        examples = expand_pairs(train_pairs, model)
        loss_fn = MultipleNegativesSymmetricRankingLoss(model)
        objective = "paired (text<->description), symmetric ranking loss"
    else:
        chunks = load_chunks()
        if not chunks:
            raise SystemExit("no training text found in %s/" % TEXT_DIR)
        examples = [(chunk, chunk) for chunk in chunks]
        loss_fn = MultipleNegativesRankingLoss(model)
        objective = "unsupervised SimCSE (dropout pairs)"
    return examples, loss_fn, objective


def finetune(epochs, batch_size, lr, max_samples, seed, base_model=BASE_MODEL,
             paired=False, holdout=0.0):
    random.seed(seed)
    torch.manual_seed(seed)

    ensure_model_writable()
    print("base model: %s" % base_model, flush=True)
    model = SentenceTransformer(base_model)  # needed up front to chunk by tokens
    device = model.device
    print("device: %s" % device, flush=True)

    # tokenize() is deprecated but functional; it returns the feature dict the
    # loss expects. Silence the one-time notice.
    warnings.filterwarnings("ignore", message=".*tokenize.*deprecated.*")

    examples, loss_fn, objective = build_examples(model, paired, holdout, seed)
    random.shuffle(examples)
    if max_samples and len(examples) > max_samples:
        examples = examples[:max_samples]
    print("objective: %s" % objective, flush=True)
    print("training on %d example pairs" % len(examples), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dummy_labels = torch.tensor([], dtype=torch.long)

    model.train()
    for epoch in range(epochs):
        random.shuffle(examples)
        running, steps = 0.0, 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            if len(batch) < 2:
                continue  # need at least one negative for the ranking loss
            texts_a = [a for a, _ in batch]
            texts_b = [b for _, b in batch]
            view_a = to_device(model.tokenize(texts_a), device)
            view_b = to_device(model.tokenize(texts_b), device)

            optimizer.zero_grad()
            loss = loss_fn([view_a, view_b], dummy_labels)
            loss.backward()
            optimizer.step()

            running += loss.item()
            steps += 1
            if steps % 5 == 0:
                print("  epoch %d step %d/%d  loss %.4f"
                      % (epoch + 1, steps, len(examples) // batch_size + 1,
                         running / steps), flush=True)
        print("epoch %d done — mean loss %.4f"
              % (epoch + 1, running / max(steps, 1)), flush=True)
        # Checkpoint after every epoch so a long CPU run that dies partway
        # still leaves a usable model on disk.
        model.save(MODEL_DIR)
        model.train()  # save() flips to eval(); resume training mode
        print("  checkpoint saved to %s/ (through epoch %d)"
              % (MODEL_DIR, epoch + 1), flush=True)

    model.eval()
    model.save(MODEL_DIR)
    print("saved fine-tuned model to %s/" % MODEL_DIR)
    print("next: regenerate vectors with `python embedding.py`, "
          "then score with `python evaluate.py`")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a custom embedding model on the text/ listings."
    )
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="sentence-transformers base to fine-tune")
    parser.add_argument("--paired", action="store_true",
                        help="train on (text, description) pairs from %s/ "
                             "instead of unsupervised SimCSE" % DESC_DIR)
    parser.add_argument("--holdout", type=float, default=0.0,
                        help="fraction of VINs to reserve for evaluate.py "
                             "(paired mode only, e.g. 0.1)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=8000,
                        help="cap on training examples (0 = use all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    finetune(args.epochs, args.batch_size, args.lr, args.max_samples,
             args.seed, args.base_model, args.paired, args.holdout)


if __name__ == "__main__":
    main()
