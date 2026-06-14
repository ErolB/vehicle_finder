import os
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from playwright.sync_api import sync_playwright

# The inventory is rendered client-side, so we drive a real browser and read the
# DOM after the vehicle cards have been injected. The card's title link carries
# both the vehicle name (its text) and the detail-page href.
CARD_TITLE = ".vehicle-card-title a"


class Scraper:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_links(self, max_pages=None):
        vehicles = []
        seen = set()
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()

            # page 1 (= ?start=0)
            page.goto(self.base_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector(CARD_TITLE, timeout=30000)
            self._collect(page, vehicles, seen)

            # the results are paginated with ?start=N (N = 0, page_size, 2*page_size, ...)
            max_start, step = self._pagination(page)
            pages_done = 1
            for start in range(step, max_start + 1, step):
                if max_pages and pages_done >= max_pages:
                    break
                page.goto(self._with_start(start), wait_until="domcontentloaded", timeout=60000)
                page.wait_for_selector(CARD_TITLE, timeout=30000)
                self._collect(page, vehicles, seen)
                pages_done += 1

            browser.close()
        return vehicles

    def _collect(self, page, vehicles, seen):
        """Append (vehicle_name, link) tuples from the current page (deduped by link)."""
        pairs = page.eval_on_selector_all(
            CARD_TITLE, "els => els.map(e => [e.textContent.trim(), e.href])")
        for name, href in pairs:
            href = href.split("#")[0]
            if self._is_vehicle(href) and href not in seen:
                seen.add(href)
                vehicles.append((name, href))

    def _pagination(self, page):
        """Return (max_start, step) read from the pagination's ?start= values."""
        hrefs = page.eval_on_selector_all("ul.pagination a", "els => els.map(e => e.href)")
        starts = []
        for href in hrefs:
            query = parse_qs(urlparse(href).query)
            if "start" in query:
                starts.append(int(query["start"][0]))
        if not starts:
            return 0, 1
        max_start = max(starts)
        # the smallest positive start is the page size (i.e. the "page 2" offset)
        step = min((s for s in starts if s > 0), default=max_start or 1)
        return max_start, step

    def _with_start(self, start):
        parts = urlparse(self.base_url)
        query = parse_qs(parts.query)
        query["start"] = [str(start)]
        return urlunparse(parts._replace(query=urlencode(query, doseq=True)))

    @staticmethod
    def _is_vehicle(href):
        return href.endswith(".htm") and ("/new/" in href or "/used/" in href or "/certified" in href)

    @staticmethod
    def _extract_vin(page):
        """Read the VIN from the detail page's data layer, with a regex fallback."""
        vin = page.evaluate(
            "() => { try { return window.DDC.dataLayer['vehicle-details'][0].vin || null; }"
            " catch (e) { return null; } }")
        if not vin:
            match = re.search(r'"vin"\s*:\s*"([A-HJ-NPR-Z0-9]{17})"', page.content())
            vin = match.group(1) if match else None
        return vin

    def scrape_page(self, vehicle_name, url):
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            vin = self._extract_vin(page)
            page_html = page.content()
            browser.close()
        os.makedirs("html", exist_ok=True)
        # filename is the VIN; fall back to the name only if the VIN can't be read
        filename = vin or vehicle_name.replace("/", "_").replace("\\", "_")
        with open("html/%s.html" % filename, "w", encoding="utf-8") as html_file:
            html_file.write(page_html)
    
    def scrape(self):
        for vehicle_name, link in self.get_links():
            self.scrape_page(vehicle_name, link)


if __name__ == "__main__":
    scraper = Scraper(base_url="https://www.beckchryslerdodgejeep.com/used-inventory/index.htm")
    print(scraper.scrape())
