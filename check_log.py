import os
import openai
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def extract_pdf(file_path):
    import PyPDF2
    reader = PyPDF2.PdfReader(file_path)
    list_text = [page.extract_text() for page in reader.pages]
    text = "".join(list_text)
    return text


key = "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"

os.environ["OPENAI_API_KEY"] = key
openai.api_key = key

pdf_file_path = "test.pdf"
text = extract_pdf(pdf_file_path)

from llama_index import Document, GPTVectorStoreIndex
documents = [Document(text)]
index = GPTVectorStoreIndex.from_documents(documents)

question = "Is Toronto the fastest growing city in North America? Format answer in paragraphs and create a bullet point for each paragraph"

query_engine = index.as_query_engine()
response = query_engine.query(question)
print(response)