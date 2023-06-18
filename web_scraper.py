import requests
from typing import Union
from bs4 import BeautifulSoup as BS


def get_repository_links(user_url: str) -> Union[list, None]:
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


def get_all_file_links(repo_link: str, file_links: list) -> Union[list, None]:
    print("Current file links length:", len(file_links))
    response = requests.get(repo_link, headers={'User-Agent': "Chrome/51.0.2704.106"})
    if response.status_code != 200:
        print("Error Occurred: Response Code: ", response.status_code)
        return None
    soup = BS(response.content, 'html.parser')
    # if this is the main repository page, then our desired links have class 'js-navigation-open' and 'Link--primary'
    # else links inside each folder contains only class 'Link--primary'
    if len(repo_link.split('/')) <= 5:  # the main page url has 5 parts when split by '/', other pages have > 5 parts
        anchor_tags = soup.select('a.js-navigation-open.Link--primary')
    else:
        anchor_tags = soup.select('a.Link--primary')
    # Process the anchor tags
    for a_tag in anchor_tags:
        # Link__StyledLink-sc-14289xe-0 dSlCya
        # if a_tag['class'][0] == 'Link__StyledLink-sc-14289xe-0' and a_tag['class'][1] == 'dSlCya'
        if len(a_tag.attrs['class']) <=2:
            href = a_tag.attrs['href']
            link = 'https://github.com' + href
            # if link contains 'blob' in url then it is a link to a file, hence add this link to 'file_links'
            if 'blob' in link:
                file_links.append(link)
            else:
                get_all_file_links(link, file_links)


if __name__ == "__main__":
    # url = input("Enter URL:") + "?tab=repositories"
    url = 'https://github.com/hhhrrrttt222111' + "?tab=repositories"
    repositories = get_repository_links(url)
    print(f"Found {len(repositories)} repositories")
    if len(repositories) > 0:
        all_file_links = list()
        # for repository in repositories:
        print(f"Found {len(all_file_links)} files")
