import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from config import google_api_key
# Load environment variables from .env file
os.environ['GOOGLE_API_KEY'] = google_api_key
# Get Google API key from environment variables
os.getenv("GOOGLE_API_KEY")
# Configure Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def create_conversational_chain():
    prompt_template = """
    YOU ARE AN HELPFULL AI ASSISTANT. YOUR TASK IS TO ANSWER ALL THE QUESTIONS ASKED BY USER BASED ON THE GIVEN CONTEXT.
\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = create_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

# Main function
def main():
    st.set_page_config("AI Assistant")
    st.header("AI Assistant")
    user_question = st.text_input("Ask any question from the data source: ")
    if user_question:
        process_user_input(user_question)
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdf(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("All the data from PDFs has been added to database")

if __name__ == "__main__":
    main()