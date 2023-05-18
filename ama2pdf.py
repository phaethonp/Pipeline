

# import subprocess

# install dependencies
# subprocess.run(['pip', 'install', '-r', 'requirements.txt'])


# import libraries
from llama_index import Document, ServiceContext, LLMPredictor, StorageContext
from llama_index import GPTVectorStoreIndex, GPTListIndex, GPTKeywordTableIndex, GPTTreeIndex
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index import load_index_from_storage
from langchain import OpenAI
from enum import Enum
import PyPDF2
# import gdown
import os


def require_chatgptkey():
    """ Make user input chatgpt key"""
    # code
    # for now just use available
    key = "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"
    return key


def extract_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    list_text = [page.extract_text() for page in reader.pages]
    text = "".join(list_text)
    return text

def create_index_from_text(text, index_type):
    index_types = [GPTVectorStoreIndex, GPTListIndex, GPTKeywordTableIndex, GPTKnowledgeGraphIndex, GPTTreeIndex]
    methods = [create_vector_index_from_text, create_list_index_from_text, create_tree_index_from_text,
               create_kwtable_index_from_text, create_kg_index_from_text]
    n = index_types.index(index_type)
    index = methods[n](text)
    return index

def create_vector_index_from_text(text):
    documents = [Document(text)]
    index = GPTVectorStoreIndex.from_documents(documents)
    return index

def create_list_index_from_text(text):
    documents = [Document(text)]
    index = GPTListIndex.from_documents(documents)
    return index

def create_tree_index_from_text(text):
    documents = [Document(text)]
    index = GPTTreeIndex.from_documents(documents)
    return index

def create_kwtable_index_from_text(text):
    documents = [Document(text)]
    index = GPTKeywordTableIndex.from_documents(documents)
    return index

def create_kg_index_from_text(text):
    documents = [Document(text)]
    index = GPTKnowledgeGraphIndex.from_documents(documents)
    return index

def get_pdf_file():
    save_path = os.path.join(os.getcwd(), "test.pdf")

    return save_path

def choose_index_type():
    index_types = [GPTVectorStoreIndex, GPTListIndex, GPTKeywordTableIndex, GPTKnowledgeGraphIndex, GPTTreeIndex]
    dirs = ["vector_index", "list_index", "kwtable_index", "graph_index", "tree_index"]
    for i, index_type in enumerate(index_types):
        print(i, ": ", index_type.__name__)
    n = -1
    while n not in range(len(dirs)):
        n = int(input("Please choose the method you want to index: "))
    return index_types[n], dirs[n]

def save_indexes(index, dir):
    parent_dir = os.getcwd()
    persist_dir = os.path.join(parent_dir, dir)

    try:
        os.makedirs(persist_dir)
    except:
        pass

    index.storage_context.persist(persist_dir=persist_dir)
    return persist_dir

def query(query_engine):
    while 1:
        question = input("""\nWhat do you want to ask? (type "exit" if you don't)\n""")
        if question == "exit":
            break

        response = query_engine.query(question)
        print(response)


def main():
    key = require_chatgptkey()
    os.environ["OPENAI_API_KEY"] = key

    pdf_file_path = get_pdf_file()
    text = extract_pdf(pdf_file_path)

    index_type, dir = choose_index_type()
    index = create_index_from_text(text, index_type)

    # save indexes 
    persist_dir = save_indexes(index, dir)

    # load indexes
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    

    query_engine = index.as_query_engine()

    query(query_engine)


main()
