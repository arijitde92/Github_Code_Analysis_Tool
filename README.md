# Github_Code_Analysis_Tool
A Python-based tool which, when given a GitHub user's URL, returns the most technically complex and challenging repository from that user's profile. The tool will use GPT and LangChain to assess each repository individually before determining the most technically challenging one. 

## Guide for Installation
Before running the tool run the requirements.txt file to install all the packages required for this tool.

`command: pip install -r requirements.txt`

## Tokens
You need to generate three tokens.
1. A GitHub token. [see here](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
2. An OpenAI token. [see here](https://openai.com/blog/openai-api)
3. An ActiveLoop. [see here](https://docs.activeloop.ai/storage-and-credentials/user-authentication)

After creating the tokens, create a file called *tokens.txt* and write as shown below-<br>
github,"your token here"<br>
openai,"your token here"<br>
activeloop,"your token here"

# Run the Tool
To run the GitHub Code Analysis Tool you need to run the following command:<br>
`flask run`<br>
You will be provided with the localhost link which you can open to visit the webpage.
Once the webpage is opened you will see a textbox asking for GitHub profile url.<br>
All you need to do now is, type the GitHub user's URL and you will be provided with the most technically complex and challenging repository of that user's profile.
Although you have to wait a bit for the app to process all the repositories.