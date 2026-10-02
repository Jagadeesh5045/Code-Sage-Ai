"""HTML parsing utilities for the web scraper."""

import re
from urllib.parse import urljoin

def parse_html(html):
    """Parse HTML and return structured sections."""
    sections = []
    # Split by major block elements
    blocks = re.split(r"<(?:div|section|article|main)[^>]*>", html)
    for block in blocks:
        text = strip_tags(block).strip()
        if len(text) > 50:
            sections.append(text)
    return sections

def extract_links(html, base_url):
    """Extract all href links from HTML, resolved to absolute URLs."""
    links = []
    for match in re.finditer(r'href=["\'](.*?)["\']\', html):
        href = match.group(1).strip()
        if href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        if absolute.startswith("http"):
            links.append(absolute)
    return list(set(links))

def extract_text(html):
    """Extract visible text from HTML, removing tags and scripts."""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = strip_tags(text)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_tags(html):
    """Remove all HTML tags from a string."""
    return re.sub(r"<[^>]+>", " ", html)
