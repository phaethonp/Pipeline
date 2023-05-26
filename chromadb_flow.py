# import subprocess

# install dependencies
# subprocess.run(['pip', 'install', 'chromadb', 'llama-index', 'openai', 'PyPDF2', 'transformers'])

from llama_index import SimpleDirectoryReader, GPTListIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.readers.schema.base import Document
import chromadb
import PyPDF2
import os
import openai
from chromadb.config import Settings
from llama_index.readers.chroma import ChromaReader

def require_chatgptkey():
    """ Make user input chatgpt key"""
    # code
    # for now just use available
    key = "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"
    return key


def connect_to_chromadb():
    chroma_client = chromadb.Client(Settings(chroma_api_impl="rest",
                                        chroma_server_host="18.168.149.193",
                                        chroma_server_http_port="8000"
                                    ))
    return chroma_client


def extract_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    list_text = [page.extract_text() for page in reader.pages]
    text = "".join(list_text)
    return text


def create_vector_index_from_text(text, chroma_collection):
    documents = [Document(text)]

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # GPTVectorStoreIndex uses llama embed
    from llama_index import GPTVectorStoreIndex
    index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

# def create_list_index_from_text(text, chroma_collection):
#     documents = [Document(text)]
    
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # GPTListIndex not uses llama embed
#     index = GPTListIndex.from_documents(documents, storage_context=storage_context)
#     return index

# def create_tree_index_from_text(text, chroma_collection):
#     documents = [Document(text)]
    
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # GPTTreeIndex uses llama embed
#     from llama_index import GPTTreeIndex
#     index = GPTTreeIndex.from_documents(documents, storage_context=storage_context)
#     return index

# def create_kwtable_index_from_text(text, chroma_collection):
#     documents = [Document(text)]

#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # GPTKeywordTableIndex uses llama embed
#     from llama_index import GPTKeywordTableIndex
#     index = GPTKeywordTableIndex.from_documents(documents, storage_context=storage_context)
#     return index

# def create_kg_index_from_text(text, chroma_collection):
#     documents = [Document(text)]

#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # GPTKnowledgeGraphIndex uses llama embed
#     from llama_index import GPTKnowledgeGraphIndex
#     index = GPTKnowledgeGraphIndex.from_documents(documents, storage_context=storage_context)
#     return index


def get_collection_reader(collection_name):
    reader = ChromaReader(
        host="18.168.149.193",
        port=8000,
        collection_name=collection_name
    )
    return reader


def query_from_chromadb(reader, question, embedding_model="text-embedding-ada-002"):
    query_vector = openai.Embedding.create(
        model=embedding_model, 
        input=question
    )

    embeddings = query_vector['data'][0]['embedding']

    results = reader._collection.query(query_embeddings=embeddings,
                                    include=["documents", "embeddings", "metadatas"],
                                    n_results=10)
    return results


def get_document_nodes_from_query_results(results):
    document_nodes = []
    for result in zip(
        results["ids"],
        results["documents"],
        results["embeddings"],
        results["metadatas"],
    ):
        document = Document(
            doc_id=result[0][0],
            text=result[1][0],
            embedding=result[2][0],
            extra_info=result[3][0],
        )
        document_nodes.append(document)
    return document_nodes


def get_response(document_nodes, question):
    index = GPTListIndex.from_documents(document_nodes)

    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    return response

if __name__ == "__main__":
    key = require_chatgptkey()
    os.environ["OPENAI_API_KEY"] = key
    openai.api_key = key


    # save index data of "test.pdf" to chromadb "test" collection unless it is already saved
    name = "test"
    pdf_file_path = f"{name}.pdf"
    question = "summarize the article"

    try:
        # query if index data already created
        reader = get_collection_reader(name)
        results = query_from_chromadb(reader, question)
        document_nodes = get_document_nodes_from_query_results(results)
        response = get_response(document_nodes, question)
        print(response)
    except Exception:
        # if not, create index data
        text = extract_pdf(pdf_file_path)
        chroma_client = connect_to_chromadb()
        chroma_collection = chroma_client.create_collection(name)
        index = create_vector_index_from_text(text, chroma_collection)