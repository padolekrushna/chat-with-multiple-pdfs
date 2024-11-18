
from flask import Flask, request, render_template, jsonify
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file.save(f"./uploads/{file.filename}")

    # Load PDF and process
    loader = PyPDFDirectoryLoader("./uploads")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff",
        llm=pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B"),
    )

    # Example query
    query = request.form.get("query", "Provide a summary of the documents.")
    answer = qa_chain.run(query)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
