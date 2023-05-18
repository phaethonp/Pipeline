from llama_index import Document, ServiceContext, LLMPredictor, StorageContext
from llama_index import GPTVectorStoreIndex
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


def create_index_from_text(text):
    documents = [Document(text)]
    index = GPTVectorStoreIndex.from_documents(documents)
    return index


def get_pdf_file():
    save_path = os.path.join(os.getcwd(), "B2B_TA_Example_MSC_Retail_Agent_Agreement.pdf")

    return save_path


def save_indexes(index):
    parent_dir = os.getcwd()
    persist_dir = os.path.join(parent_dir, "index_dir")

    try:
        os.makedirs(persist_dir)
    except:
        pass

    index.storage_context.persist(persist_dir=persist_dir)
    return persist_dir


def query(query_engine, question):
    response = query_engine.query(question)
    print(response.response)
    return response.response
