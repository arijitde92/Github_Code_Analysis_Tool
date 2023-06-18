# Imports
from flask import Flask, render_template, request
import validators
from web_scraper import get_repository_links
from Document_Loading import *
import os
import shutil
import requests
import base64
from typing import Union, Tuple, List
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import GitLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

# Initialize flask app
app = Flask(__name__)


# http://localhost:5000
@app.route('/', methods=['GET', 'POST'])
def index():
    valid_url = True
    url = ""
    if request.method == 'POST':
        url = request.form['repo']
        if validators.url(url):
            valid_url = True
        else:
            valid_url = False
        render_template('index.html', valid_url=valid_url)

    if valid_url:
        if os.path.exists(os.path.join(VECTOR_DB_PATH, "git_test")):
            print("Removing already existing vector store and creating new one")
            shutil.rmtree(os.path.join(VECTOR_DB_PATH, "git_test"))
        repositories = get_repository_links(url+"?tab=repositories")
        print(f"Found {len(repositories)} repositories")
        if len(repositories) > 0:
            embeddings = OpenAIEmbeddings(disallowed_special=())
            db = DeepLake(dataset_path=os.path.join(VECTOR_DB_PATH, "git_test"), embedding_function=embeddings)
            for repo_url in tqdm(repositories):
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

                "From the repositories - HelloFlask, AD_Classification_App, chroma-langchain, Alzheimer_Classification, "
                "flasklogin, AV_Segmentation_Tool and Phantom_Generator as given in context.\nCan you briefly describe each of them?",
                "Can you rank the above repositories in terms of complexity?"
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
        return render_template('index.html', valid_url=True, results=True,
                               repositories=repositories, most_complex_repo="ABCD")
    return render_template('index.html', valid_url=valid_url)


if __name__ == "__main__":
    app.run(debug=True)
