import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sidebar Information
with st.sidebar:
    st.title('PromPDF')
    st.markdown('''
    ## About
    Upload a PDF file, and our system will analyze it. Ask any question about the document, and we'll give you precise answers.
    ''')

# App Header
st.header("PromPDF Chatbot")

# Initialize session state to hold the chat history
if "history" not in st.session_state:
    st.session_state.history = []

# PDF Uploader
pdf = st.file_uploader("Upload PDF", type='pdf')
if pdf is not None:
    # Read PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Create embeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    VectorStore = FAISS.from_texts(chunks, hf_embeddings)

    # Save embeddings to disk
    store_name = pdf.name[:-4]
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

    # Chat Interface
    user_input = st.text_input("Ask a question about the uploaded PDF:")

    if user_input:
        # Add the user question to the history
        st.session_state.history.append({"role": "user", "content": user_input})

        # Fetch relevant documents from the PDF
        docs = VectorStore.similarity_search(query=user_input, k=3)

        # Prepare the LLM (Language Model)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 1, "max_length": 1024},
        )

        # Load QA chain
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        # Run the chain with the documents and user question
        response = chain.run(input_documents=docs, question=user_input)

        # Add the bot's response to the history
        st.session_state.history.append({"role": "bot", "content": response.strip()})

    # Display the conversation history
    if st.session_state.history:
        for chat in st.session_state.history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**Bot:** {chat['content']}")

