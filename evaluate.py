import os
import glob
import argparse

import numpy as np

# Reuse the exact text-embedding the search index uses, so the score reflects
# real retrieval quality rather than a separate code path.
from embedding import Embedder, find_boilerplate

TEXT_DIR = "text"
DESC_DIR = "descriptions"
MODEL_DIR = "model"
HOLDOUT_FILE = os.path.join(MODEL_DIR, "holdout.txt")


def load_holdout():
    """VINs reserved by `finetune.py --holdout`, or empty for an in-sample run."""
    if not os.path.exists(HOLDOUT_FILE):
        return set()
    with open(HOLDOUT_FILE, encoding="utf-8") as holdout_file:
        return {line.strip() for line in holdout_file if line.strip()}


def main():
    parser = argparse.ArgumentParser(
        description="Score the model: does each vehicle's description retrieve "
                    "its own listing? Reports recall@k for description->listing."
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10],
                        help="recall@k cutoffs to report")
    args = parser.parse_args()

    # VINs that have both a listing and a description.
    vins, text_paths, desc_paths = [], [], []
    for desc_path in sorted(glob.glob(os.path.join(DESC_DIR, "*.txt"))):
        vin = os.path.splitext(os.path.basename(desc_path))[0]
        text_path = os.path.join(TEXT_DIR, "%s.txt" % vin)
        if os.path.exists(text_path):
            vins.append(vin)
            text_paths.append(text_path)
            desc_paths.append(desc_path)
    if not vins:
        raise SystemExit("no text+description pairs found to evaluate")

    embedder = Embedder()  # loads the trained model from model/
    boilerplate = find_boilerplate(text_paths)

    # Corpus: one listing vector per VIN, exactly as embedding.py builds the index.
    print("embedding %d listings (corpus)..." % len(vins))
    corpus = np.vstack([embedder.embed_file(p, boilerplate) for p in text_paths])
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

    # Queries: descriptions for the eval set (held-out VINs if finetune recorded
    # them, otherwise every VIN — an optimistic in-sample sanity check).
    holdout = load_holdout()
    eval_idx = [i for i, vin in enumerate(vins) if vin in holdout] or list(range(len(vins)))
    mode = "held-out" if holdout else "all (in-sample)"
    print("scoring descriptions from %d %s listings against %d listings...\n"
          % (len(eval_idx), mode, len(vins)))

    ks = sorted(args.ks)
    hits = {k: 0 for k in ks}
    reciprocal, ranks = 0.0, []
    for i in eval_idx:
        # describe.py writes several descriptions per listing (one per line);
        # each is a separate query that should still retrieve VIN i's listing.
        with open(desc_paths[i], encoding="utf-8") as desc_file:
            descriptions = [l.strip() for l in desc_file if l.strip()]
        for description in descriptions:
            query = embedder.embed_text(description)
            query = query / np.linalg.norm(query)
            scores = corpus @ query
            # rank (0-based) of this VIN's own listing among all listings
            rank = int((scores > scores[i]).sum())
            ranks.append(rank)
            reciprocal += 1.0 / (rank + 1)
            for k in ks:
                if rank < k:
                    hits[k] += 1

    n = len(ranks)
    print("recall@k (description finds its own listing):")
    for k in ks:
        print("  @%-3d %.3f" % (k, hits[k] / n))
    print("MRR         %.3f" % (reciprocal / n))
    print("median rank %d   (0 = top hit, out of %d)" % (int(np.median(ranks)), len(vins)))


if __name__ == "__main__":
    main()
