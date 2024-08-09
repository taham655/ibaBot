import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Streamlit app title
st.title("PDF Question Answering App")

# Get OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

system_prompt = """
You are a friendly assistant that only answer the questions from the provided context. If you dont know the answer just say you dont. 

Context: {context}

Question: {question}
"""

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        # Initialize ChatOpenAI
        llm = ChatOpenAI(model="gpt-4-0125-preview", api_key=openai_api_key)

        # Load, chunk and index the contents of the PDF
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Initialize embeddings and vectorstore
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        # Retrieve and generate using the relevant snippets of the PDF
        retriever = vectorstore.as_retriever()
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # User input for question
        user_question = st.text_input("Ask a question about the PDF:")

        if user_question:
            with st.spinner("Generating answer..."):
                answer = rag_chain.invoke(user_question)
                st.write("Answer:", answer)

    finally:
        # Clean up: Delete the temporary file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
            st.success("Temporary file cleaned up successfully.")

else:
    st.info("Please upload a PDF file to get started.")