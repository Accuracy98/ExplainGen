import requests
from bs4 import BeautifulSoup
import re
import json
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_all_article_links(base_url):
    page = 1
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    article_links = []

    while True:
        url = f"{base_url}?page={page}"
        try:
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch page {page}: {response.status_code}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')

            found_new_links = False
            for a_tag in soup.select('div.m-statement__quote a'):
                link = a_tag.get('href')
                if link and link.startswith('/factchecks/'):
                    full_url = "https://www.politifact.com" + link
                    if full_url not in article_links:
                        article_links.append(full_url)
                        found_new_links = True

            if not found_new_links:
                print("No more articles found, reached the last page.")
                break

            print(f"📄 Page {page} processed, moving to next page...")
            page += 1
            time.sleep(random.uniform(2, 5))

        except requests.exceptions.RequestException as e:
            print(f"⚠Error fetching page {page}: {str(e)}")
            break

    return article_links


def extract_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title_quote = None
        for quote in soup.select('div.m-statement__quote'):
            text = clean_text(quote.get_text())
            if text:
                title_quote = text
                break

        summary = None
        summary_tag = soup.select_one('h1.c-title')
        if summary_tag:
            summary = clean_text(summary_tag.get_text())

        claims = []
        main_content = soup.select_one('article.m-textblock')
        if main_content:
            for p in main_content.select('p:not(.m-statement__quote)'):
                text = clean_text(p.get_text())
                if text and text != title_quote:
                    claims.append(text)

        print(f"{url} - {title_quote if title_quote else 'No Title'}")
        return {'url': url, 'title': title_quote, 'summary': summary, 'claims': claims}

    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return {'url': url, 'title': None, 'summary': None, 'claims': []}


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\xa0\u200b]', '', text)
    return text.strip()


def save_results(data_list, filename='results.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    base_url = 'https://www.politifact.com/factchecks/list/'

    print("\nFetching all article links...")
    article_links = get_all_article_links(base_url)

    if not article_links:
        print("No articles found. Exiting...")
    else:
        print(f"Found {len(article_links)} articles.")

        print("\nExtracting article contents...")
        results = []
        for i, link in enumerate(article_links, start=1):
            print(f"[{i}/{len(article_links)}] Scraping: {link}")
            article_data = extract_content(link)
            results.append(article_data)

            save_results(results)

            time.sleep(random.uniform(2, 5))

        print("\nData extraction completed and saved as results.json")
