import os
import glob
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

# Our custom model, fine-tuned on text/ by finetune.py. Run that first.
MODEL = "model"
TEXT_DIR = "text"
EMBED_DIR = "embeddings"

# A line is treated as boilerplate (dealer nav/header/footer, identical across
# pages) when it shows up in at least this fraction of all listings. The real
# template chrome appears on ~every page, so a high bar leaves vehicle-specific
# text — including features shared by some cars — untouched.
BOILERPLATE_DF = 0.9


def document_lines(text_path):
    """Return the non-empty, stripped lines of one listing."""
    with open(text_path, encoding="utf-8") as text_file:
        return [line.strip() for line in text_file if line.strip()]


def find_boilerplate(text_paths, threshold=BOILERPLATE_DF):
    """Lines that appear in >= threshold of the listings — the shared chrome."""
    doc_freq = Counter()
    for path in text_paths:
        for line in set(document_lines(path)):  # count each line once per page
            doc_freq[line] += 1
    cutoff = threshold * len(text_paths)
    return {line for line, count in doc_freq.items() if count >= cutoff}


class Embedder:
    def __init__(self, model=MODEL):
        self.model = load_model(model)
        self.tokenizer = self.model.tokenizer
        # Leave room for the [CLS]/[SEP] tokens encode() adds to each chunk.
        self.window = self.model.max_seq_length - 2

    def embed_text(self, text):
        """Embed arbitrarily long text by mean-pooling over token-window chunks.

        The model truncates anything past max_seq_length, so a single encode()
        would only see the first chunk. Instead we split the token stream into
        windows, embed each, and average — so the whole page is represented.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = self.tokenizer.encode(" ", add_special_tokens=False)
        chunks = [
            self.tokenizer.decode(ids[start:start + self.window])
            for start in range(0, len(ids), self.window)
        ]
        vectors = self.model.encode(chunks)  # (n_chunks, dim)
        return vectors.mean(axis=0)

    def embed_file(self, text_path, boilerplate=frozenset()):
        """Return the embedding for one listing, minus the shared chrome."""
        lines = [l for l in document_lines(text_path) if l not in boilerplate]
        return self.embed_text("\n".join(lines))

    def embed(self):
        os.makedirs(EMBED_DIR, exist_ok=True)
        text_paths = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
        boilerplate = find_boilerplate(text_paths)
        print("dropping %d boilerplate lines shared across listings"
              % len(boilerplate))
        for text_path in text_paths:
            # mirror text/<VIN>.txt -> embeddings/<VIN>.npy
            vin = os.path.splitext(os.path.basename(text_path))[0]
            out_path = os.path.join(EMBED_DIR, "%s.npy" % vin)
            if os.path.exists(out_path):
                continue  # already embedded — lets the batch resume after a stop
            vector = self.embed_file(text_path, boilerplate)
            np.save(out_path, vector)
            print("embedded %s (%d dims)" % (vin, vector.shape[0]))


def load_model(model=MODEL):
    if not os.path.isdir(model):
        raise SystemExit(
            "custom model %r not found — run `python finetune.py` first" % model
        )
    return SentenceTransformer(model)


if __name__ == "__main__":
    Embedder().embed()
