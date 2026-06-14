import os
import glob
import concurrent.futures
from html.parser import HTMLParser
import anthropic
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and anything else) from a local .env file.
load_dotenv()

# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "Your task is to extract all text that a human would find relevant from an HTML page. " \
    "Exclude customer reviews and links."
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
HTML_DIR = "html"
TEXT_DIR = "text"

# When False, output the raw text parser result. When True, also pass that
# text through the API for further extraction.
USE_API = True

# Files are parsed concurrently. The API calls are independent per file, so the
# binding limit is tokens-per-minute, not CPU — keep this modest (each page is
# ~200K input tokens) and let the SDK back off on 429s.
MAX_WORKERS = 6


class _TextExtractor(HTMLParser):
    """Collect the text content of every element in the document."""

    # Tags whose contents aren't human-visible text.
    _SKIP_TAGS = {"script", "style"}
    # Tags that flow within a line of text; everything else is a block boundary.
    _INLINE_TAGS = {
        "a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "em", "i",
        "kbd", "mark", "q", "s", "small", "span", "strong", "sub", "sup",
        "time", "u", "var", "wbr",
    }

    def __init__(self):
        super().__init__()
        self._skip_depth = 0  # how deep we are inside a skipped element
        self._chunks = []

    def _boundary(self, tag):
        # Block-level tags separate text; insert a newline so words don't merge.
        if tag not in self._INLINE_TAGS:
            self._chunks.append("\n")

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        self._boundary(tag)

    def handle_startendtag(self, tag, attrs):
        # Self-closing tags (e.g. <br/>) — treat <br> as a line break.
        self._boundary(tag)

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        self._boundary(tag)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self):
        # Collapse the runs of whitespace that HTML layout leaves behind.
        lines = (line.strip() for line in "".join(self._chunks).splitlines())
        return "\n".join(line for line in lines if line)


def extract_all_text(html):
    """Return the text from all elements, excluding <script>/<style> contents."""
    parser = _TextExtractor()
    parser.feed(html)
    all_text = parser.get_text()
    #print(all_text)
    return all_text


class Parser:
    def __init__(self, model=MODEL, system_prompt=SYSTEM_PROMPT):
        # Reads ANTHROPIC_API_KEY from the environment — don't hardcode a key.
        self.client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt

    def extract_text(self, html):
        """Send one page's HTML to /v1/messages and return the extracted text."""
        kwargs = {
            "model": self.model,
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": html}],
        }
        # Only attach a system prompt once you've written one. The cache_control
        # breakpoint caches it across every file (no-op while it's blank).
        if self.system_prompt:
            kwargs["system"] = [{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]

        # Each page is ~200K tokens of input, so stream to avoid request timeouts.
        with self.client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        return "".join(b.text for b in message.content if b.type == "text")

    def parse_file(self, html_path):
        with open(html_path, encoding="utf-8") as html_file:
            text = extract_all_text(html_file.read())
        # USE_API False: stop at the text parser. True: refine it through the API.
        if USE_API:
            return self.extract_text(text)
        return text

    def _parse_one(self, html_path):
        """Parse one page and write text/<VIN>.txt. Returns (vin, chars) or None
        if it was already parsed. Runs in a worker thread."""
        # mirror html/<VIN>.html -> text/<VIN>.txt
        vin = os.path.splitext(os.path.basename(html_path))[0]
        out_path = os.path.join(TEXT_DIR, "%s.txt" % vin)
        if os.path.exists(out_path):
            return None  # already parsed — lets the batch resume after a stop
        text = self.parse_file(html_path)
        with open(out_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        return vin, len(text)

    def parse(self):
        os.makedirs(TEXT_DIR, exist_ok=True)
        html_paths = sorted(glob.glob(os.path.join(HTML_DIR, "*.html")))
        # The Anthropic client is thread-safe, and each page's call is independent
        # and I/O-bound, so a thread pool parses many at once.
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(self._parse_one, p): p for p in html_paths}
            for future in concurrent.futures.as_completed(futures):
                vin = os.path.splitext(os.path.basename(futures[future]))[0]
                try:
                    result = future.result()
                except anthropic.APIError as err:
                    print("FAILED %s: %s" % (vin, err))
                    continue
                if result:
                    print("parsed %s (%d chars)" % result)


if __name__ == "__main__":
    Parser().parse()
