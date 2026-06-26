import os
import re
import glob
import argparse
import numpy as np

# Our custom model, fine-tuned on text/ by finetune.py. Must match the model
# embedding.py used to build the vectors in EMBED_DIR.
MODEL = "model"
# Voyage model used by the --api backend (key in .env as VOYAGE_API_KEY). Must
# match the model embedding.py --api built EMBED_DIR with.
VOYAGE_MODEL = "voyage-4-large"
EMBED_DIR = "embeddings"
TEXT_DIR = "text"
HTML_DIR = "html"

# The dealership listing URL lives in the saved page's canonical <link>.
_CANONICAL_TAG = re.compile(r'<link\b[^>]*\brel="canonical"[^>]*>', re.IGNORECASE)
_HREF = re.compile(r'\bhref="([^"]+)"', re.IGNORECASE)
# New vs used is encoded in the listing URL path (.../new/... or .../used/...).
_URL_CONDITION = re.compile(r"/(new|used)/", re.IGNORECASE)
# Mileage appears in two listing layouts: a table row ("| Odometer | 69,269
# miles |") and a bold bullet ("- **Odometer:** 69,269 miles"). Both put only
# punctuation/whitespace ([:*|] and spaces) between the label and the number, so
# requiring that — rather than allowing any text — is what keeps this from
# matching the CARFAX sentence "Odometer is 6,590 miles below market average"
# (the word "is" sits between "Odometer" and the number there).
_ODOMETER = re.compile(r"Odometer[:*|\s]*([\d,]+)\s*miles", re.IGNORECASE)
# Prices sit in a Pricing block of label/amount rows, in the same two layouts as
# mileage: table cells ("| MSRP | $45,905 |") and bold bullets ("- **Price:**
# $21,000"). Match a Price/MSRP label joined to a "$" amount by punctuation only
# ([:*|] and spaces) — so the dealer-fee and trade-in dollar amounts in the
# disclaimer prose ("$995 dealer fee", "ACV of $10,000"), which have no price
# label in front, are left out. Captures (label, sign, amount).
_PRICE_ROW = re.compile(r"(MSRP|[\w ]*Price)[:*|\s]*(-?)\$\s*([\d,]+)", re.IGNORECASE)
# Label fragments that mark a discount/savings row (e.g. "Beck Yeah Price",
# "... Discount") rather than a price the customer pays.
_DISCOUNT_LABEL = re.compile(r"yeah|discount|saving|rebate", re.IGNORECASE)


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


def vehicle_mileage(vin, text_dir=TEXT_DIR):
    """Return the odometer reading (int miles) from a listing, else None.

    Reads the Overview table's odometer row. New vehicles have no such row, so
    they return None (no meaningful mileage to show). Returns None if the text
    file is missing.
    """
    path = os.path.join(text_dir, "%s.txt" % vin)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as text_file:
        match = _ODOMETER.search(text_file.read())
    return int(match.group(1).replace(",", "")) if match else None


def vehicle_price(vin, text_dir=TEXT_DIR):
    """Return (price, list_price) in whole dollars for a listing.

    price is the dealer's lowest advertised (sale) price; list_price is the MSRP
    when one is shown and it's higher than price, so the caller can strike it
    through. Both are None when the listing has no numeric price (e.g. "Price:
    Please Call"); list_price alone is None when there's no higher MSRP to show.
    Reads the Pricing rows and skips discount/savings lines and the dealer-fee /
    trade-in dollar amounts in the disclaimer prose.
    """
    path = os.path.join(text_dir, "%s.txt" % vin)
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as text_file:
        text = text_file.read()

    msrp, sale = None, []
    for label, sign, amount in _PRICE_ROW.findall(text):
        value = int(amount.replace(",", ""))
        if "MSRP" in label.upper():
            msrp = value
        elif sign == "-" or _DISCOUNT_LABEL.search(label):
            continue  # a discount/savings line, not a price the customer pays
        else:
            sale.append(value)

    # A few listings put a tiny delta in the sale-price cell (e.g. "$280 (above
    # MSRP after incentives)"). When an MSRP is known, drop sale figures
    # implausibly far below it so that quirk can't masquerade as the price; real
    # new-car discounts stay well above half of MSRP.
    if msrp is not None:
        sale = [v for v in sale if v >= msrp / 2]

    if sale:
        price = min(sale)  # the dealer's best advertised price
    else:
        price, msrp = msrp, None  # only an MSRP listed — show it as the price
    show_msrp = msrp if (price is not None and msrp is not None and msrp > price) else None
    return price, show_msrp


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


def _in_range(value, low, high):
    """True if value falls within the optional [low, high] bounds.

    An unknown value (None — e.g. a "Please Call" price or a new car's missing
    odometer) passes: a best-effort numeric pre-filter shouldn't drop a listing
    just because the figure couldn't be parsed. Bounds are inclusive; either end
    may be None to leave that side open.
    """
    if value is None:
        return True
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


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
    # Imported lazily so the --api backend doesn't require sentence-transformers.
    from sentence_transformers import SentenceTransformer

    if not os.path.isdir(model):
        raise SystemExit(
            "custom model %r not found — run `python finetune.py` first" % model
        )
    return SentenceTransformer(model)


class LocalEncoder:
    """Encode the query with the local fine-tuned model (the default)."""

    def __init__(self, model=MODEL):
        self.model = load_model(model)

    def encode(self, text):
        return self.model.encode(text)


class VoyageEncoder:
    """Encode the query with the Voyage API (--api).

    Reads VOYAGE_API_KEY from .env. Uses input_type="query" — the asymmetric
    counterpart to the "document" embeddings embedding.py --api wrote to disk.
    """

    def __init__(self, model=VOYAGE_MODEL):
        # Imported lazily so local-only users don't need the voyageai package.
        import voyageai
        from dotenv import load_dotenv

        load_dotenv()  # populate VOYAGE_API_KEY from .env
        if not os.environ.get("VOYAGE_API_KEY"):
            raise SystemExit("set VOYAGE_API_KEY in .env to use --api")
        # The client reads VOYAGE_API_KEY from the environment.
        self.client = voyageai.Client()
        self.model = model

    def encode(self, text):
        result = self.client.embed(
            [text or " "], model=self.model, input_type="query"
        )
        return np.array(result.embeddings[0], dtype=np.float32)


class Searcher:
    def __init__(self, encoder=None, embed_dir=EMBED_DIR):
        self.encoder = encoder or LocalEncoder()
        self.embed_dir = embed_dir

    def _load_embeddings(self, keywords=None, conditions=None,
                         price_range=(None, None), mileage_range=(None, None)):
        """Return (vins, matrix) for the candidate embeddings/<VIN>.npy files.

        Listings are filtered before the vector search, so ranking happens over
        the survivors only. `keywords` keeps listings whose text contains all of
        them; `conditions` (e.g. {"new", "used"}) keeps the matching conditions;
        `price_range` and `mileage_range` are (low, high) dollar/mile bounds that
        keep listings whose price/odometer falls within them (see _in_range).
        """
        min_price, max_price = price_range
        min_mileage, max_mileage = mileage_range
        check_price = min_price is not None or max_price is not None
        check_mileage = min_mileage is not None or max_mileage is not None

        vins, vectors = [], []
        for path in sorted(glob.glob(os.path.join(self.embed_dir, "*.npy"))):
            vin = os.path.splitext(os.path.basename(path))[0]
            if conditions and vehicle_condition(vin) not in conditions:
                continue
            if keywords and not matches_keywords(vin, keywords):
                continue
            if check_price and not _in_range(
                vehicle_price(vin)[0], min_price, max_price
            ):
                continue
            if check_mileage and not _in_range(
                vehicle_mileage(vin), min_mileage, max_mileage
            ):
                continue
            vins.append(vin)
            vectors.append(np.load(path))
        if not vins:
            if keywords:
                raise SystemExit(
                    "no listings contain all keywords: %s" % ", ".join(keywords)
                )
            if check_price or check_mileage:
                raise SystemExit("no vehicles match the price/mileage filters")
            if conditions:
                raise SystemExit(
                    "no %s vehicles found" % " or ".join(sorted(conditions))
                )
            raise SystemExit("no embeddings found in %s/" % self.embed_dir)
        return vins, np.vstack(vectors)

    def search(self, prompt, top_k=5, keywords=None, conditions=None,
               price_range=(None, None), mileage_range=(None, None)):
        """Return the top_k (VIN, score) pairs closest to the prompt."""
        vins, matrix = self._load_embeddings(
            keywords, conditions, price_range, mileage_range
        )
        query = self.encoder.encode(prompt)

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
    parser.add_argument(
        "--api", action="store_true",
        help="encode the query with the Voyage API (%s) instead of the local "
             "model; EMBED_DIR must have been built with embedding.py --api too"
             % VOYAGE_MODEL,
    )
    args = parser.parse_args()

    encoder = VoyageEncoder() if args.api else LocalEncoder()
    searcher = Searcher(encoder)
    for vin, score in searcher.search(args.prompt, args.top_k, args.keywords):
        print("%s\t%.4f\t%s" % (vin, score, vehicle_name(vin)))


if __name__ == "__main__":
    main()
