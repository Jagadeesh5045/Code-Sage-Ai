"""Web scraper for extracting structured data from web pages."""

import re
import time
import urllib.request
from urllib.parse import urljoin, urlparse
from html_parser import parse_html, extract_links, extract_text

class WebScraper:
    """Configurable web scraper with rate limiting and content extraction."""

    def __init__(self, base_url, delay=1.0):
        self.base_url = base_url
        self.delay = delay
        self.visited = set()
        self.results = []

    def scrape_page(self, url):
        """Scrape a single page and extract content."""
        if url in self.visited:
            return None
        self.visited.add(url)

        try:
            html = self._fetch(url)
            title = self._extract_title(html)
            text = extract_text(html)
            links = extract_links(html, url)
            headings = self._extract_headings(html)

            result = {
                "url": url,
                "title": title,
                "text": text[:5000],
                "links": links,
                "headings": headings,
            }
            self.results.append(result)
            time.sleep(self.delay)
            return result

        except Exception as e:
            return {"url": url, "error": str(e)}

    def crawl(self, max_pages=10):
        """Crawl starting from base URL, following links up to max_pages."""
        queue = [self.base_url]
        while queue and len(self.visited) < max_pages:
            url = queue.pop(0)
            result = self.scrape_page(url)
            if result and "links" in result:
                for link in result["links"]:
                    if self._is_same_domain(link) and link not in self.visited:
                        queue.append(link)
        return self.results

    def _fetch(self, url):
        """Fetch HTML content from a URL."""
        req = urllib.request.Request(url, headers={"User-Agent": "CodeSage-Scraper/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _extract_title(self, html):
        """Extract page title from HTML."""
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_headings(self, html):
        """Extract all headings (h1-h6) from HTML."""
        headings = []
        for match in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", html, re.IGNORECASE | re.DOTALL):
            level = int(match.group(1))
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if text:
                headings.append({"level": level, "text": text})
        return headings

    def _is_same_domain(self, url):
        """Check if URL belongs to the same domain as base URL."""
        base_domain = urlparse(self.base_url).netloc
        url_domain = urlparse(url).netloc
        return base_domain == url_domain
