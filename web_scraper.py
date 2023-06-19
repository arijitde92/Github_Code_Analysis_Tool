# Importing required libraries
import requests
from typing import Union
from bs4 import BeautifulSoup as BS


def get_repository_links(user_url: str) -> Union[list, None]:
    """
    The get_repository_links function is designed to get the list of GitHub repository links.
    Parameters
    ----------
    user_url : str
       Github url.
    Returns
    ----------
    repo_links : list
        List of repositories.
    """
    repo_links = list()
    user_name = user_url.split('?')[0].split('/')[-1]
    response = requests.get(user_url, headers={'User-Agent': "Chrome/51.0.2704.106"})
    if response.status_code != 200:
        print("Error Occurred: Response Code: ", response.status_code)
        return None
    html_content = response.content
    soup = BS(html_content, 'html.parser')
    repo_headings = soup.select('h3.wb-break-all')
    for repo_heading in repo_headings:
        repo_name = repo_heading.a.attrs["href"].split('/')[-1]
        link = 'https://github.com/' + user_name + "/" + repo_name
        repo_links.append(link)
    pages = soup.find_all(attrs={"class": "next_page"})
    if len(pages) > 0:
        # find the next page link if <a href> exists
        if pages[0].name == 'a':
            next_page_link = 'https://github.com' + pages[0].attrs['href']
            repo_links = repo_links + get_repository_links(next_page_link)
    return repo_links


# Main function
if __name__ == "__main__":
    # url = input("Enter URL:") + "?tab=repositories"
    url = 'https://github.com/hhhrrrttt222111' + "?tab=repositories"
    repositories = get_repository_links(url)
    print(f"Found {len(repositories)} repositories")
    if len(repositories) > 0:
        all_file_links = list()
        # for repository in repositories:
        print(f"Found {len(all_file_links)} files")
