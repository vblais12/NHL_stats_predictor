import os
import requests
import time
from src.config import CACHE_DIR
import hashlib
import logging

logger = logging.getLogger(__name__)


# Takes URL --> file name for caching
def url_to_file(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest() + '.html'

# Function to get html from a page. Checks cache first, else fetches page then caches
def get_page(url, headers=None):

    file = url_to_file(url)
    filename = os.path.join(CACHE_DIR, file)

    if os.path.exists(filename):
        logger.info(f"{filename} already cached, getting page...")
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    logger.info(f"File not cached --> Fetching {url}...")
    try:
        response = requests.get(url, headers=headers)
        time.sleep(3.1)
        response.raise_for_status()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
            logger.info(f"File cached. Excellent. Moving right along")
        return response.text
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


