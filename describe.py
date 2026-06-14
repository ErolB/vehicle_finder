import os
import glob
import concurrent.futures
import anthropic
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and anything else) from a local .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# Write your instructions here. Each vehicle's listing text (text/<VIN>.txt) is
# sent as the user message; this system prompt tells Claude how to turn it into
# a short description. The script refuses to run while this is blank.
SYSTEM_PROMPT = "Generate a short description of this vehicle (under 50 words). Highlight features " \
    "that a potential customer would find relevant. This will be used to train another model."
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"  # matches parser.py; bump to claude-opus-4-8 for higher quality
TEXT_DIR = "text"
DESC_DIR = "descriptions"
MAX_TOKENS = 512  # short descriptions; generated tokens are all you pay for
MAX_WORKERS = 6   # calls are independent and I/O-bound; the SDK backs off on 429s


class Describer:
    def __init__(self, model=MODEL, system_prompt=SYSTEM_PROMPT):
        if not system_prompt.strip():
            raise SystemExit("write SYSTEM_PROMPT in describe.py before running")
        # Reads ANTHROPIC_API_KEY from the environment — don't hardcode a key.
        self.client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt

    def describe_text(self, text):
        """Send one listing's text to /v1/messages and return the description."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            # The system prompt is identical across every file; the cache_control
            # breakpoint caches it (a no-op below the model's min cacheable size).
            system=[{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": text}],
        )
        return "".join(b.text for b in message.content if b.type == "text").strip()

    def describe_file(self, text_path):
        """Describe one listing and write descriptions/<VIN>.txt. Returns
        (vin, chars) or None if already done. Runs in a worker thread."""
        # mirror text/<VIN>.txt -> descriptions/<VIN>.txt
        vin = os.path.splitext(os.path.basename(text_path))[0]
        out_path = os.path.join(DESC_DIR, "%s.txt" % vin)
        if os.path.exists(out_path):
            return None  # already described — lets the batch resume after a stop
        with open(text_path, encoding="utf-8") as text_file:
            text = text_file.read()
        description = self.describe_text(text)
        with open(out_path, "w", encoding="utf-8") as desc_file:
            desc_file.write(description)
        return vin, len(description)

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
                    print("described %s (%d chars)" % result)


if __name__ == "__main__":
    Describer().describe()
