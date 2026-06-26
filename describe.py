import os
import re
import glob
import random
import concurrent.futures
import anthropic
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and anything else) from a local .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# Two-phase prompting. For each listing we first ask Claude to pull out the
# vehicle's notable features (FEATURE_PROMPT), then ask it to write several
# short descriptions, each one highlighting a *random combination* of those
# features (DESCRIPTION_PROMPT). The random subsets are chosen in Python so the
# augmentation is reproducible per VIN. The script refuses to run while either
# prompt is blank.
FEATURE_PROMPT = (
    "Extract this vehicle's notable, customer-relevant features from the "
    "listing. Output one short feature phrase per line (e.g. 'leather seats', "
    "'low mileage', 'all-wheel drive'). No numbering, no commentary, no blank "
    "lines. Use simple and clear language."
)
DESCRIPTION_PROMPT = (
    "You write short descriptions for new and used vehicles. Given a listing and a "
    "set of feature groups, write one description (under 20 words) per group. "
    "Each description must highlight that group's features, stay accurate to "
    "the listing, and should be clear and concise. These will be "
    "used to train another model.\n"
    "Output exactly one description per group, in order, one per line. No "
    "numbering, no blank lines, no extra commentary."
)
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"  # matches parser.py; bump to claude-opus-4-8 for higher quality
TEXT_DIR = "text"
DESC_DIR = "descriptions"
MAX_TOKENS = 512  # short descriptions; generated tokens are all you pay for
MAX_WORKERS = 6   # calls are independent and I/O-bound; the SDK backs off on 429s

# How many descriptions to produce per listing, and how many features each one
# combines (a random count in [MIN_FEATURES, MAX_FEATURES] per description).
NUM_DESCRIPTIONS = 5
MIN_FEATURES = 2
MAX_FEATURES = 4

# Strip a leading list marker ("1. ", "- ", "* ") the model sometimes adds back
# despite the instruction, so each stored line is just the description text.
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def _clean_lines(raw):
    """Split a model reply into clean, non-empty, single-line entries."""
    lines = []
    for line in raw.splitlines():
        line = _LIST_MARKER.sub("", line.strip())
        line = " ".join(line.split())  # collapse stray whitespace to one line
        if line:
            lines.append(line)
    return lines


class Describer:
    def __init__(self, model=MODEL, feature_prompt=FEATURE_PROMPT,
                 description_prompt=DESCRIPTION_PROMPT):
        if not feature_prompt.strip() or not description_prompt.strip():
            raise SystemExit("write FEATURE_PROMPT and DESCRIPTION_PROMPT in "
                             "describe.py before running")
        # Reads ANTHROPIC_API_KEY from the environment — don't hardcode a key.
        self.client = anthropic.Anthropic()
        self.model = model
        self.feature_prompt = feature_prompt
        self.description_prompt = description_prompt

    def _call(self, system_prompt, user_text):
        """One /v1/messages round-trip; returns the concatenated text reply."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            # The system prompt is identical across every file; the cache_control
            # breakpoint caches it (a no-op below the model's min cacheable size).
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_text}],
        )
        return "".join(b.text for b in message.content if b.type == "text").strip()

    def extract_features(self, text):
        """Return the listing's notable features as a list of short phrases."""
        return _clean_lines(self._call(self.feature_prompt, text))

    def describe_combos(self, text, combos):
        """Write one description per feature group (a list of feature lists)."""
        groups = "\n".join(
            "Group %d: %s" % (i + 1, "; ".join(combo))
            for i, combo in enumerate(combos)
        )
        user_text = (
            "Vehicle listing:\n%s\n\n"
            "Write one description for each feature group below.\n\n%s"
            % (text, groups)
        )
        lines = _clean_lines(self._call(self.description_prompt, user_text))
        # Keep at most one description per group; the model occasionally returns
        # a different count, so we align to the groups we asked for.
        return lines[:len(combos)]

    def describe_text(self, text, rng):
        """Produce NUM_DESCRIPTIONS descriptions from random feature subsets.

        Falls back to a single whole-listing description when the listing yields
        too few features to combine.
        """
        features = self.extract_features(text)
        if len(features) < MIN_FEATURES:
            # Not enough features to form combinations — one plain description.
            return self.describe_combos(text, [features or ["this vehicle"]])
        combos = []
        for _ in range(NUM_DESCRIPTIONS):
            size = min(rng.randint(MIN_FEATURES, MAX_FEATURES), len(features))
            combos.append(rng.sample(features, size))
        return self.describe_combos(text, combos)

    def describe_file(self, text_path):
        """Describe one listing and write descriptions/<VIN>.txt (one
        description per line). Returns (vin, n_descriptions) or None if already
        done. Runs in a worker thread."""
        # mirror text/<VIN>.txt -> descriptions/<VIN>.txt
        vin = os.path.splitext(os.path.basename(text_path))[0]
        out_path = os.path.join(DESC_DIR, "%s.txt" % vin)
        if os.path.exists(out_path):
            return None  # already described — lets the batch resume after a stop
        with open(text_path, encoding="utf-8") as text_file:
            text = text_file.read()
        # Seed per-VIN so the random feature combinations are reproducible.
        descriptions = self.describe_text(text, random.Random(vin))
        if not descriptions:
            return None
        with open(out_path, "w", encoding="utf-8") as desc_file:
            desc_file.write("\n".join(descriptions))
        return vin, len(descriptions)

    def describe(self):
        os.makedirs(DESC_DIR, exist_ok=True)
        text_paths = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
        # The Anthropic client is thread-safe and each call is independent and
        # I/O-bound, so a thread pool describes many listings at once.
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(self.describe_file, p): p for p in text_paths}
            for future in concurrent.futures.as_completed(futures):
                vin = os.path.splitext(os.path.basename(futures[future]))[0]
                try:
                    result = future.result()
                except anthropic.APIError as err:
                    print("FAILED %s: %s" % (vin, err))
                    continue
                if result:
                    print("described %s (%d descriptions)" % result)


if __name__ == "__main__":
    Describer().describe()
