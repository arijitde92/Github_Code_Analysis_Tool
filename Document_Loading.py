import os
import shutil
import requests
import fnmatch
import base64
from typing import Union, Tuple, List
from git import Repo
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import GitLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from tqdm import tqdm

REPO_DIR = "./Clone"
VECTOR_DB_PATH = "./VectorStore"
os.environ['OPENAI_API_KEY'] = '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'
os.environ[
    'ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4NjkzMTI2OSwiZXhwIjoxNjkwODE5MTk4fQ' \
                          '.eyJpZCI6ImFyaWppdGRlOTIifQ.KpP_d-aul8YlgEBZEbIaZsouKaHvWJQvrzAQZU8' \
                          '-QQuoMl0xUUrUP2E32NQDeyWRl7Cz0KuNj4DHkhbpH4cH5w'
os.environ['GITHUB_TOKEN'] = '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'
NON_TEXT_EXTENSIONS = [".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi", ".mp3", ".mp4", ".ind", ".indt", ".indd",
                       ".mov", ".png", ".gif", ".webp", ".tiff", ".tif", ".psd", ".raw", ".bmp", ".dib", ".heif",
                       ".heic", ".eps", ".ai", ".svg", ".mkv", ".avi", ".pyc", ".pt", ".pth", ".pb", ".h5", ".ckpt",
                       ".rar", ".zip", ".tar", ".iso", ".dat", ".ico", ".docx", ".pptx", ".xlsx", ".xls", ".ppt",
                       ".doc", ".gitignore", ".nii"]
PROGRAMMING_LANGUAGES = {".py": "python", ".cpp": "C++", ".c": "C", ".h": "C or C++ header", ".java": "Java",
                         ".r": "R", ".html": "Hyper Text Markup Language (HTML)", ".css": "Cascading Style Sheet",
                         ".js": "JavaScript", ".ipynb": "Python Jupyter Notebook", ".md": "Markdown", ".php": "PHP",
                         ".cs": "C Sharp", ".ts": "TypeScript", ".sql": "SQL", ".tex": "Latex"}


def parse_github_url(url: str) -> Tuple[str, str]:
    """
    The parse_github_url(url) function is designed to extract the owner and repository name from a given GitHub
    repository URL.
    :param url: github url
    :return: the owner and repo names
    """
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo


def get_files_from_github_repo(owner: str, repo: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    elif response.status_code == 404:
        # Might be the default main branch is called 'main' and not 'master', hence changing the url
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json()
            return content["tree"]
        else:
            raise ValueError(f"Error fetching repo contents: {response.status_code}")
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")


def fetch_contents(files: list):
    contents = []
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json"
    }

    def is_non_text_file(filename, extensions):
        return any(filename.endswith(e) for e in extensions)

    def get_programming_language(filename):
        ext = filename.split('.')[-1]
        lang = PROGRAMMING_LANGUAGES.get("."+ext)
        if lang is None:
            return "Unknown"
        else:
            return lang

    for file in tqdm(files):
        if file["type"] == "blob" and not is_non_text_file(file["path"], NON_TEXT_EXTENSIONS):
            response = requests.get(file["url"], headers=headers)
            if response.status_code == 200:
                content = response.json()["content"]
                try:
                    decoded_content = base64.b64decode(content).decode('utf-8')
                    repository_name = file['url'].split('/')[5]
                    contents.append(Document(page_content=decoded_content, metadata={"source": file['path'],
                                                                                     "repository": repository_name,
                                                                                     "programming_language:": get_programming_language(file['path'])}))
                except Exception as e:
                    print(f"Decoding error {e}")
                    print("For file: ", file['path'])
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return contents


def extract_repo_data(link: str) -> list[Document]:
    repo_path = os.path.join(REPO_DIR, link.split('/')[-1])
    if not os.path.exists(repo_path):
        # if the repository does not already exist in Clone directory, clone and download it to repo_path
        repo = Repo.clone_from(link, to_path=repo_path)
    else:
        # if repository exists, then load from the directory instead of cloning
        repo = Repo(repo_path)
    branch = repo.head.reference
    loader = GitLoader(repo_path=repo_path, branch=branch)
    data = loader.load()
    return data


def delete_repo(repo_link: str):
    repo_path = os.path.join(REPO_DIR, repo_link.split('/')[-1])
    try:
        shutil.rmtree(repo_path)
    except Exception as e:
        print(e)


def get_chunks_from_files(files, chunk_size=1024, chunk_overlap=20):
    source_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = fetch_contents(files)
    for source in documents:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadate=source.metadata))
    return source_chunks


if __name__ == "__main__":
    repo_links = ["https://github.com/arijitde92/Alzheimer_Classification"
                  "https://github.com/ISHITA1234/HelloFlask",
                  "https://github.com/arijitde92/AD_Classification_App",
                  "https://github.com/hwchase17/chroma-langchain",
                  "https://github.com/ISHITA1234/flasklogin",
                  "https://github.com/arijitde92/AV_Segmentation_Tool",
                  "https://github.com/arijitde92/Phantom_Generator"
                  ]
    embeddings = OpenAIEmbeddings(disallowed_special=())
    username = "arijitde92"  # replace with your username from app.activeloop.ai
    db = DeepLake(dataset_path=os.path.join(VECTOR_DB_PATH, "git_test"), embedding_function=embeddings)
    for repo_url in tqdm(repo_links):
        print("\nGetting files from ", repo_url)
        github_username, repo_name = parse_github_url(repo_url)
        files = get_files_from_github_repo(github_username, repo_name)
        print("Splitting the files into chunks")
        data_chunks = get_chunks_from_files(files)
        print("Adding extracted chunks to Vector Store at", VECTOR_DB_PATH)
        db.add_documents(data_chunks)

    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20
    retriever.search_kwargs["reduce_k_below_max_tokens"] = True

    # model_name = "gpt-4"
    # llm = OpenAI(model_name=model_name, temperature=0)
    # qa_chain = load_qa_chain(llm, chain_type=
    # qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)  # 'ada' 'gpt-3.5-turbo' 'gpt-4',
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    questions = [
        # "We are trying to analyse the repositories given in the context. Can you name the repositories and can you "
        # "tell me what programming languages are used in these repositories?",
        "What does train.py do?",
        "Analyize the complexity of train.py."
        # "From the repositories - HelloFlask, AD_Classification_App, chroma-langchain, Alzheimer_Classification, "
        # "flasklogin, AV_Segmentation_Tool and Phantom_Generator as given in context.\nCan you briefly describe each of them?",
        # "Can you rank the above repositories in terms of complexity?"
        # "What classes are derived from the Chain class?",
        # "What classes and functions in the ./langchain/utilities/ folder are not covered by unit tests?",
        # "What one improvement do you propose in code in relation to the class herarchy for the Chain class?",
    ]
    chat_history = []

    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
