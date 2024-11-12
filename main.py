import os
import streamlit as st
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import faiss  # Import FAISS library directly for index handling

load_dotenv()

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_file_path = "faiss_index"

main_placefolder = st.empty()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading ... Started")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placefolder.text("Text Splitter ... Started")
    docs = text_splitter.split_documents(data)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector Started Building")
    time.sleep(2)

    # Save the FAISS index and document store details
    faiss.write_index(vectorstore_openai.index, index_file_path)
    with open("docstore.pkl", "wb") as f:
        pickle.dump(vectorstore_openai.docstore, f)
    with open("index_to_docstore_id.pkl", "wb") as f:
        pickle.dump(vectorstore_openai.index_to_docstore_id, f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(index_file_path) and os.path.exists("docstore.pkl") and os.path.exists("index_to_docstore_id.pkl"):
        # Load the FAISS index, docstore, and index_to_docstore_id mapping
        index = faiss.read_index(index_file_path)
        with open("docstore.pkl", "rb") as f:
            docstore = pickle.load(f)
        with open("index_to_docstore_id.pkl", "rb") as f:
            index_to_docstore_id = pickle.load(f)

        # Define embeddings again for reconstructing the vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id=index_to_docstore_id)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display result
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
