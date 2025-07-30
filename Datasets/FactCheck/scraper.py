import requests
from bs4 import BeautifulSoup
import json
import time

# Main page URL
base_url = "https://www.factcheck.org/covid-misconceptions/"

# Set headers to avoid being blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Send a request to the main page
response = requests.get(base_url, headers=headers)
response.raise_for_status()

# Parse the main page HTML using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all "Fact Checks" section headers
fact_checks_headers = soup.find_all('h3', class_='sans', text=lambda t: t and 'Fact Checks' in t)

# Initialize data storage
data = []

# Iterate through each "Fact Checks" section
for fact_checks_header in fact_checks_headers:
    print("📝 Processing a 'Fact Checks' section...")

    # Find the next sibling <ul> element and iterate through all <li> elements containing <a> tags
    ul_element = fact_checks_header.find_next_sibling('ul')
    if ul_element:
        fact_check_links = ul_element.find_all('li')

        for li in fact_check_links:
            link = li.find('a')
            if link:
                # Extract link title and URL
                title = link.get('title', 'No Title')
                url = link.get('href')
                print(f"Scraping: {title} -> {url}")
                try:
                    # Send a request to the linked page
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    page_soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract SciCheck Digest or Quick Take content
                    digest_header = page_soup.find('h2', string='SciCheck Digest')
                    quick_take_header = page_soup.find('h2', string='Quick Take')
                    section_type = ""
                    summary_content = ""

                    if digest_header:  # Check for SciCheck Digest
                        section_type = "SciCheck Digest"
                        content_elements = []
                        for sibling in digest_header.find_next_siblings():
                            if sibling.name == 'hr':  # Stop when reaching the separator
                                break
                            if sibling.name == 'p':  # Extract paragraph content only
                                for a_tag in sibling.find_all('a'):  # Replace <a> tags with plain text
                                    a_tag.replace_with(a_tag.get_text())
                                content_elements.append(sibling.get_text(" ", strip=True))
                        summary_content = " ".join(content_elements)

                    elif quick_take_header:  # Check for Quick Take
                        section_type = "Quick Take"
                        content_elements = []
                        for sibling in quick_take_header.find_next_siblings():
                            if sibling.name == 'hr':  # Stop when reaching the separator
                                break
                            if sibling.name == 'p':  # Extract paragraph content only
                                for a_tag in sibling.find_all('a'):  # Replace <a> tags with plain text
                                    a_tag.replace_with(a_tag.get_text())
                                content_elements.append(sibling.get_text(" ", strip=True))
                        summary_content = " ".join(content_elements)

                    # Extract Full Story content
                    full_story_header = page_soup.find('h2', string='Full Story')
                    full_story_content = ""
                    if full_story_header:
                        content_elements = []
                        for sibling in full_story_header.find_next_siblings():
                            if sibling.name == 'hr':  # Stop when reaching the separator
                                break
                            if sibling.name == 'p':  # Extract paragraph content only
                                for a_tag in sibling.find_all('a'):  # Replace <a> tags with plain text
                                    a_tag.replace_with(a_tag.get_text())
                                content_elements.append(sibling.get_text(" ", strip=True))
                        full_story_content = " ".join(content_elements)

                    # Save the extracted data only if summary and full story exist
                    if summary_content and full_story_content:
                        data.append({
                            'url': url,
                            'title': title,
                            'section_type': section_type,
                            'summary': summary_content.strip(),
                            'full_story': full_story_content.strip()
                        })
                        print(f"Successfully scraped: {title}")
                    else:
                        print(f"⚠Skipped: {title} (missing required sections)")
                    time.sleep(1)

                except Exception as e:
                    print(f"Scraping failed: {url}, error: {e}")
    else:
        print("Could not find the 'ul' list for this 'Fact Checks' section!")

# Save the data to a JSON file
output_file = 'covid-misconceptions_data.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Data scraping completed and saved to {output_file}!")
