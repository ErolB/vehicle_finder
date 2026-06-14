import os
import re
import glob
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

# Our custom model, fine-tuned on text/ by finetune.py. Must match the model
# embedding.py used to build the vectors in EMBED_DIR.
MODEL = "model"
EMBED_DIR = "embeddings"
TEXT_DIR = "text"
HTML_DIR = "html"

# The dealership listing URL lives in the saved page's canonical <link>.
_CANONICAL_TAG = re.compile(r'<link\b[^>]*\brel="canonical"[^>]*>', re.IGNORECASE)
_HREF = re.compile(r'\bhref="([^"]+)"', re.IGNORECASE)
# New vs used is encoded in the listing URL path (.../new/... or .../used/...).
_URL_CONDITION = re.compile(r"/(new|used)/", re.IGNORECASE)


def _listing_title(vin, text_dir=TEXT_DIR):
    """The first line of a listing's text, normalized.

    The API-refined output (USE_API in parser.py) prefixes the title with a
    markdown heading, e.g. "# Used 2020 Jeep Wrangler ...", so strip any leading
    '#' before parsing. Returns "" if the text file is missing.
    """
    path = os.path.join(text_dir, "%s.txt" % vin)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as text_file:
        return text_file.readline().lstrip("#").strip()


def vehicle_name(vin, text_dir=TEXT_DIR):
    """Pull the vehicle name from the listing's title (first line of its text).

    Titles look like "Used 2020 Jeep Wrangler Unlimited Sport For Sale | ..." or
    "Used 2020 Jeep Wrangler Unlimited Sport - Beck Chrysler Dodge Jeep"; the
    name is everything before the " For Sale" / " - <dealer>" suffix. Falls back
    to the VIN if the text file is missing.
    """
    title = _listing_title(vin, text_dir)
    if not title:
        return vin
    name = title.split(" For Sale")[0].split(" - ")[0].strip()
    return name or vin


def vehicle_condition(vin, text_dir=TEXT_DIR, html_dir=HTML_DIR):
    """Return "new" or "used" for a listing, else None.

    Prefer the title's first word (cheap). The API-refined output sometimes drops
    the New/Used prefix, so fall back to the /new/ or /used/ segment of the
    dealership canonical URL, which is authoritative regardless of title wording.
    """
    first_word = _listing_title(vin, text_dir).split(" ", 1)[0].lower()
    if first_word in ("new", "used"):
        return first_word
    url = listing_url(vin, html_dir)
    match = _URL_CONDITION.search(url) if url else None
    return match.group(1).lower() if match else None


def listing_url(vin, html_dir=HTML_DIR):
    """Return the dealership listing URL from the saved page's canonical link."""
    path = os.path.join(html_dir, "%s.html" % vin)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as html_file:
        head = html_file.read(200000)  # canonical sits in <head>, near the top
    tag = _CANONICAL_TAG.search(head)
    if not tag:
        return None
    href = _HREF.search(tag.group(0))
    return href.group(1) if href else None


def matches_keywords(vin, keywords, text_dir=TEXT_DIR):
    """True if the listing's text contains every keyword as a whole word.

    Matching is case-insensitive and word-boundary aware, so "red" won't match
    "covered". The boundaries are word/non-word transitions (lookarounds rather
    than \\b), which keeps keywords with punctuation like "4x4" or "F-150" working.
    """
    path = os.path.join(text_dir, "%s.txt" % vin)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as text_file:
        text = text_file.read().lower()
    return all(
        re.search(r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)", text)
        for keyword in keywords
    )


def load_model(model=MODEL):
    if not os.path.isdir(model):
        raise SystemExit(
            "custom model %r not found — run `python finetune.py` first" % model
        )
    return SentenceTransformer(model)


class Searcher:
    def __init__(self, model=MODEL, embed_dir=EMBED_DIR):
        self.model = load_model(model)
        self.embed_dir = embed_dir

    def _load_embeddings(self, keywords=None, conditions=None):
        """Return (vins, matrix) for the candidate embeddings/<VIN>.npy files.

        Listings are filtered before the vector search, so ranking happens over
        the survivors only. `keywords` keeps listings whose text contains all of
        them; `conditions` (e.g. {"new", "used"}) keeps the matching conditions.
        """
        vins, vectors = [], []
        for path in sorted(glob.glob(os.path.join(self.embed_dir, "*.npy"))):
            vin = os.path.splitext(os.path.basename(path))[0]
            if conditions and vehicle_condition(vin) not in conditions:
                continue
            if keywords and not matches_keywords(vin, keywords):
                continue
            vins.append(vin)
            vectors.append(np.load(path))
        if not vins:
            if keywords:
                raise SystemExit(
                    "no listings contain all keywords: %s" % ", ".join(keywords)
                )
            if conditions:
                raise SystemExit(
                    "no %s vehicles found" % " or ".join(sorted(conditions))
                )
            raise SystemExit("no embeddings found in %s/" % self.embed_dir)
        return vins, np.vstack(vectors)

    def search(self, prompt, top_k=5, keywords=None, conditions=None):
        """Return the top_k (VIN, score) pairs closest to the prompt."""
        vins, matrix = self._load_embeddings(keywords, conditions)
        query = self.model.encode(prompt)

        # Cosine similarity: normalize both sides, then dot product.
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        query = query / np.linalg.norm(query)
        scores = matrix @ query

        order = np.argsort(scores)[::-1][:top_k]
        return [(vins[i], float(scores[i])) for i in order]


def main():
    parser = argparse.ArgumentParser(
        description="Find the VINs whose listing best matches a prompt."
    )
    parser.add_argument("prompt", help="what you're looking for, in plain English")
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="how many VINs to return"
    )
    parser.add_argument(
        "-w", "--keywords", nargs="+", metavar="KEYWORD",
        help="only rank vehicles whose listing text contains ALL these keywords",
    )
    args = parser.parse_args()

    for vin, score in Searcher().search(args.prompt, args.top_k, args.keywords):
        print("%s\t%.4f\t%s" % (vin, score, vehicle_name(vin)))


if __name__ == "__main__":
    main()
