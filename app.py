from flask import Flask, render_template, jsonify, request
from src.helper import load_embeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
import ollama

ollama.pull('llama3.1')

app = Flask(__name__)

embeddings = load_embeddings()

persist_directory = 'medical-chatbot/db'
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}
llm = OllamaLLM(model="llama3.1")

retriever = vectordb.as_retriever(search_kwargs={'k':2})

qa = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa.invoke({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)