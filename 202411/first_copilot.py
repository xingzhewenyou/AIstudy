import requests
from bs4 import BeautifulSoup


def fetch_wikipedia_page(page_title):
    url = f"https://en.wikipedia.org/wiki/{page_title}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None


def main():
    page_title = "Python_(programming_language)"
    soup = fetch_wikipedia_page(page_title)

    if soup:
        # Extract the title of the page
        title = soup.find('h1', {'id': 'firstHeading'}).text
        print(f"Title: {title}")

        # Extract the first non-empty paragraph of the page
        paragraphs = soup.find_all('p')
        first_paragraph = next((p.text for p in paragraphs if p.text.strip()), "No content found")
        print(f"First paragraph: {first_paragraph}")


if __name__ == "__main__":
    main()
