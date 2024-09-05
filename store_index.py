from src.helper import load_pdf, text_split, load_embeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

extracted_data = load_pdf("medical-chatbot/data")
text_chunks = text_split(extracted_data)
embeddings = load_embeddings()

persist_directory = 'medical-chatbot/db'
vectordb = Chroma.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)
vectordb = None
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)