# Imports
from Document_Loading import *
from flask import Flask, render_template, request
import validators
import sys

# Initialize flask app
app = Flask(__name__)

QA_dict = dict()


@app.route('/', methods=['GET', 'POST'])
def index():
    valid_url = ''
    if request.method == 'POST':
        url = request.form['repo']
        if validators.url(url):
            valid_url = True
        else:
            valid_url = False
        render_template('index.html', valid_url=valid_url)

        # If the url provided by the user is valid then only proceed further
        if valid_url:
            db_store_retrieve(url)
            repositories = get_repository_links(url)
            print(QA_dict)
            final_retirever = create_questions_db(QA_dict)
            final_questions = [
                # "Which repositories are we discussing about?",
                # "Tell me a brief summary of each repository",
                "Compare the complexities of the repositories in the context and say which is the most technically "
                "complex repository for an entry level junior developer fresh out of college? Mention the repository name"
            ]
            final_answers = ask_gpt(final_retirever, final_questions)
            print("Conclusion:")
            print(final_answers[-1][-1])
            return render_template('index.html', valid_url=True, results=True,
                                   repositories=repositories, most_complex_repo=final_answers[-1][-1])
    return render_template('index.html', valid_url=valid_url)


def db_store_retrieve(url: str):
    """
    This function creates vector store databases for the repositories available in the url. Then it prompts the GPT with
    questions and stores the questions and responses by the GPT which will be further used by another GPT model to
    compare code complexity.

    Parameters
    ----------
    url: str
        The github profile URL

    Returns
    -------
        None
    """
    repositories = sorted(get_repository_links(url + "?tab=repositories"))
    print(f"Found {len(repositories)} repositories")
    if len(repositories) > 0:
        embeddings = OpenAIEmbeddings(disallowed_special=())
        for repo_url in repositories:
            _, repository_name = parse_github_url(repo_url)
            repo_retriever = create_repo_db(repo_url, embeddings)
            repo_questions = [
                f"The code snippets in your context are from a single repository called {repository_name}. How "
                f"difficult or technically complex are the code snippets in the context, for an entry level junior "
                f"developer?"
            ]
            chats = ask_gpt(repo_retriever, repo_questions)
            print("Conversation")
            print(chats)
            QA_dict[repository_name] = chats


if __name__ == "__main__":
    sys.executable = sys.executable.replace('\\App', '\\..\\..\\App')
    app.run(debug=True)
