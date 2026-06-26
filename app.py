"""Flask GUI for search.py — semantic vehicle search with a keyword pre-filter."""
import glob
import math
import os
import re

from flask import Flask, render_template, request

from search import (
    Searcher, LocalEncoder, VoyageEncoder, EMBED_DIR,
    vehicle_name, vehicle_mileage, vehicle_price, listing_url,
)

app = Flask(__name__)

# Load the model once at startup; the Searcher holds the SentenceTransformer,
# which is expensive to construct, so we reuse a single instance per process.
searcher = Searcher(LocalEncoder())


def _inventory_caps():
    """Largest price and mileage in the searchable inventory, rounded up.

    The range sliders run from 0 to these caps, so the upper handle reaches every
    real listing. Computed once at import over the same VINs search ranks (those
    with an embeddings/<VIN>.npy), reading the cached text once each. Listings
    with no parseable figure (e.g. "Please Call", or a new car's blank odometer)
    simply don't contribute a maximum.
    """
    prices, mileages = [], []
    for path in glob.glob(os.path.join(EMBED_DIR, "*.npy")):
        vin = os.path.splitext(os.path.basename(path))[0]
        price = vehicle_price(vin)[0]
        if price is not None:
            prices.append(price)
        mileage = vehicle_mileage(vin)
        if mileage is not None:
            mileages.append(mileage)
    round_up = lambda value, step: int(math.ceil(value / step) * step)
    price_cap = round_up(max(prices), 1000) if prices else 100000
    mileage_cap = round_up(max(mileages), 5000) if mileages else 200000
    return price_cap, mileage_cap


PRICE_CAP, MILEAGE_CAP = _inventory_caps()

# The Voyage-backed Searcher is built on first use (and then reused), so the app
# still starts without a VOYAGE_API_KEY unless someone flips the API toggle.
_voyage_searcher = None


def get_searcher(use_api):
    """Return the Searcher for the chosen backend.

    Local is ready at startup; Voyage is constructed lazily. VoyageEncoder()
    raises SystemExit if VOYAGE_API_KEY is missing — index() catches that and
    shows the message rather than crashing the request.
    """
    global _voyage_searcher
    if not use_api:
        return searcher
    if _voyage_searcher is None:
        _voyage_searcher = Searcher(VoyageEncoder())
    return _voyage_searcher

# A keyword is either a quoted phrase (kept whole, e.g. "Apple CarPlay") or a
# run of non-space, non-comma characters. Spaces and commas separate keywords.
_KEYWORD = re.compile(r'"([^"]*)"|\'([^\']*)\'|([^,\s]+)')


def parse_amount(name):
    """Read a non-negative integer bound (price/mileage) from the query string.

    Returns None when the field is blank, non-numeric, or negative, which leaves
    that side of the range open. Commas in the typed number are tolerated.
    """
    raw = (request.args.get(name) or "").strip().replace(",", "")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def parse_keywords(raw):
    """Split a free-text keywords field into a clean list of keywords/phrases.

    Quotes let a multi-word phrase count as one keyword; otherwise tokens are
    split on whitespace or commas.
    """
    keywords = []
    for match in _KEYWORD.finditer(raw or ""):
        phrase, single, word = match.groups()
        keyword = (phrase or single or word).strip()
        if keyword:
            keywords.append(keyword)
    return keywords


@app.route("/")
def index():
    prompt = (request.args.get("q") or "").strip()
    keywords = parse_keywords(request.args.get("keywords"))
    try:
        top_k = max(1, min(50, int(request.args.get("k", 10))))
    except ValueError:
        top_k = 10

    # New/Used checkboxes. An empty selection means "no restriction" (both), so
    # unchecking everything can't strand the user with zero possible results.
    selected = {c for c in request.args.getlist("condition") if c in ("new", "used")}
    conditions = selected or None

    # Price/mileage range pre-filter: narrow the candidate listings before the
    # semantic search, the same way keywords and condition do. Any bound left
    # blank stays open.
    min_price = parse_amount("min_price")
    max_price = parse_amount("max_price")
    min_mileage = parse_amount("min_mileage")
    max_mileage = parse_amount("max_mileage")

    # Embed the query with the Voyage API instead of the local model. The vectors
    # in embeddings/ must have been built with the matching backend (embedding.py
    # --api), or the query and document spaces won't line up.
    use_api = request.args.get("api") == "1"

    results, error = [], None
    if prompt:
        try:
            hits = get_searcher(use_api).search(
                prompt, top_k, keywords or None, conditions,
                price_range=(min_price, max_price),
                mileage_range=(min_mileage, max_mileage),
            )
            for i, (vin, score) in enumerate(hits, start=1):
                price, list_price = vehicle_price(vin)
                results.append({
                    "rank": i,
                    "vin": vin,
                    "name": vehicle_name(vin),
                    "url": listing_url(vin),
                    "mileage": vehicle_mileage(vin),
                    "price": price,
                    "list_price": list_price,
                    "score": score,
                })
        except SystemExit as exc:
            # search()/VoyageEncoder exit (missing key, no keyword matches, no
            # embeddings) — show the message instead of crashing the request.
            error = str(exc)
        except ValueError:
            # Query vector dim != stored vectors: the embeddings/ were built with
            # the other backend. Point the user at the cause.
            error = (
                "Query embedding doesn't match the stored vectors. The "
                "embeddings/ were built with the %s backend — rebuild them with "
                "the matching embedding.py run, or untick this option."
                % ("local model" if use_api else "Voyage API")
            )

    return render_template(
        "index.html",
        prompt=prompt,
        keywords=" ".join(keywords),
        top_k=top_k,
        # With an empty selection (initial load or all unchecked) both show
        # checked, matching the "no restriction" filter behavior above.
        new_checked=("new" in selected) or not selected,
        used_checked=("used" in selected) or not selected,
        # Slider extents, plus the handle positions to restore. A bound left open
        # (None) sits at its slider end — 0 for the low handle, the cap for high.
        price_cap=PRICE_CAP,
        mileage_cap=MILEAGE_CAP,
        min_price=0 if min_price is None else min_price,
        max_price=PRICE_CAP if max_price is None else max_price,
        min_mileage=0 if min_mileage is None else min_mileage,
        max_mileage=MILEAGE_CAP if max_mileage is None else max_mileage,
        api_checked=use_api,
        results=results,
        error=error,
        searched=bool(prompt),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
