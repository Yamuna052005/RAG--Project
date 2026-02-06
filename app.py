import os
import streamlit as st

from dotenv import load_dotenv  
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not found in .env file.")

def read_pdfs(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text
    return all_text
def split_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
def get_prompt_and_llm():
    prompt_template = """
    You are a helpful AI assistant.
    verbose control:
    "Provide the answer has to be in bullet points.
    and each bullet point must not be more than 80 words."
    
    Answer the question using ONLY the context below.
    If the answer is not present , say exactly:
    "The answer is not available in the provided context."
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3
        
    )
    return prompt, llm
def answer_question(question):
    vector_store = load_faiss_index()
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt, llm = get_prompt_and_llm()  
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)  
    return response.content
def main():
    st.set_page_config(page_title="Chat with PDF - RAG Demo")
    st.header("📄 Chat with PDF using Gemini")

    question = st.text_input("Ask a question from the uploaded PDFs")

    if question:
        answer = answer_question(question)
        st.subheader("Answer")
        st.write(answer)

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_files:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Reading and indexing PDFs..."):
                raw_text = read_pdfs(pdf_files)
                chunks = split_into_chunks(raw_text)
                build_faiss_index(chunks)

            st.success("PDFs processed and indexed successfully!")


if __name__ == "__main__":
    main()