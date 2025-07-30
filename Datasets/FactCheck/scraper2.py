import requests
from bs4 import BeautifulSoup
import json
import time

# Starting URL for the first page
base_url = "https://www.factcheck.org/scicheck/"

# Set headers to avoid being blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Initialize data storage
data = []

def extract_page_data(url):
    """Extract SciCheck Digest, Quick Take, and Full Story from a single article page."""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract SciCheck Digest or Quick Take content
        digest_header = soup.find('h2', string='SciCheck Digest')
        quick_take_header = soup.find('h2', string='Quick Take')
        summary_content = ""
        section_type = ""

        if digest_header:  # Check for SciCheck Digest
            section_type = "SciCheck Digest"
            for sibling in digest_header.find_next_siblings():
                if sibling.name == 'hr':  # Stop when reaching the separator
                    break
                if sibling.name == 'p':
                    for a_tag in sibling.find_all('a'):  # Replace <a> tags with plain text
                        a_tag.replace_with(a_tag.get_text())
                    summary_content += sibling.get_text(" ", strip=True) + " "

        elif quick_take_header:  # Check for Quick Take
            section_type = "Quick Take"
            for sibling in quick_take_header.find_next_siblings():
                if sibling.name == 'hr':  # Stop when reaching the separator
                    break
                if sibling.name == 'p':
                    for a_tag in sibling.find_all('a'):  # Replace <a> tags with plain text
                        a_tag.replace_with(a_tag.get_text())
                    summary_content += sibling.get_text(" ", strip=True) + " "

        # Extract Full Story content
        full_story_header = soup.find('h2', string='Full Story')
        full_story_content = ""
        if full_story_header:
            for sibling in full_story_header.find_next_siblings():
                if sibling.name == 'hr':  # Stop when reaching the separator
                    break
                if sibling.name == 'p':
                    for a_tag in sibling.find_all('a'):  # Replace <a> tags with plain text
                        a_tag.replace_with(a_tag.get_text())
                    full_story_content += sibling.get_text(" ", strip=True) + " "

        # Save content only if at least one section and Full Story are available
        if summary_content.strip() and full_story_content.strip():
            return section_type, summary_content.strip(), full_story_content.strip()
        else:
            return None, None, None
    except Exception as e:
        print(f"Failed to scrape: {url}, error: {e}")
        return None, None, None

def scrape_all_pages(start_url):
    """Scrape all pages starting from the base URL."""
    current_url = start_url
    while current_url:
        print(f"Scraping page: {current_url}")
        response = requests.get(current_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all articles on the current page
        articles = soup.find_all('article', class_='post')
        for article in articles:
            h3 = article.find('h3', class_='entry-title')
            if h3:
                link = h3.find('a')
                if link and link['href']:
                    article_url = link['href']
                    title = link.get_text(strip=True)
                    print(f"Scraping article: {title} -> {article_url}")

                    # Scrape article details
                    section_type, summary, full_story = extract_page_data(article_url)
                    if summary and full_story:
                        data.append({
                            'url': article_url,
                            'title': title,
                            'section_type': section_type,
                            'summary': summary,
                            'full_story': full_story
                        })
                        print(f"Successfully scraped: {title}")
                    else:
                        print(f"Skipped: {title} (missing required sections)")
                    time.sleep(1)  # Avoid overloading the server

        # Find the next page link
        next_page_li = soup.find('li', class_='page-item-next')
        if next_page_li:
            next_page = next_page_li.find('a', href=True)
            if next_page:
                current_url = next_page['href']
            else:
                current_url = None  # No more pages
        else:
            current_url = None  # Pagination not found, stop

# Start scraping from the first page
scrape_all_pages(base_url)

# Save the data to a JSON file
output_file = 'scicheck_data.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Data scraping completed and saved to {output_file}!")
