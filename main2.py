import os
import streamlit as st
import pickle
import time
import csv
import json
import random
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import faiss  # Import FAISS library directly for index handling

load_dotenv()

# Streamlit UI Setup
st.title("End-to-End News Research Tool with Fine-tuning")
st.sidebar.title("Input URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to start processing
process_url_clicked = st.sidebar.button("Process URLs")
generate_pairs_clicked = st.sidebar.button("Generate Q&A Pairs")
fine_tune_clicked = st.sidebar.button("Fine-tune the Model")
index_file_path = "faiss_index"
qa_csv_file = "qa_dataset.csv"
qa_jsonl_file = "qa_dataset.jsonl"
scraped_data_file = "scraped_data.txt"

main_placefolder = st.empty()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

# Step 1: Scrape Data from URLs
if process_url_clicked:
    try:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placefolder.text("Data Loading ... Started")
        data = loader.load()

        # Save raw data to a text file
        with open(scraped_data_file, "w", encoding="utf-8") as f:
            for doc in data:
                f.write(doc.page_content + "\n\n")
        main_placefolder.text(f"Raw data saved to {scraped_data_file}")

        # Split data for embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )

        main_placefolder.text("Text Splitting ... Started")
        docs = text_splitter.split_documents(data)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placefolder.text("Embedding Vector Started Building")
        time.sleep(2)

        # Save FAISS index and related details
        faiss.write_index(vectorstore_openai.index, index_file_path)
        with open("docstore.pkl", "wb") as f:
            pickle.dump(vectorstore_openai.docstore, f)
        with open("index_to_docstore_id.pkl", "wb") as f:
            pickle.dump(vectorstore_openai.index_to_docstore_id, f)

        main_placefolder.text("URLs processed successfully!")
    except Exception as e:
        main_placefolder.error(f"Error processing URLs: {str(e)}")

# Step 2: Generate Q&A Pairs
if generate_pairs_clicked:
    try:
        # Generate Q&A pairs
        def generate_pairs(input_file, output_csv):
            example_questions = [
                "What is the main topic of this paragraph?",
                "Explain the key insights mentioned here.",
                "Summarize this content briefly.",
                "What is the conclusion of this section?"
            ]

            with open(input_file, "r", encoding="utf-8") as f:
                content = f.readlines()

            qa_pairs = []
            for paragraph in content:
                if len(paragraph.strip()) > 20:  # Skip short lines
                    question = random.choice(example_questions)
                    answer = paragraph.strip()
                    qa_pairs.append((question, answer))

            with open(output_csv, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["prompt", "completion"])
                writer.writerows(qa_pairs)

            return len(qa_pairs)

        count = generate_pairs(scraped_data_file, qa_csv_file)
        main_placefolder.text(f"Generated {count} Q&A pairs and saved to {qa_csv_file}")
    except Exception as e:
        main_placefolder.error(f"Error generating Q&A pairs: {str(e)}")

# Step 3: Convert Q&A CSV to JSONL
if fine_tune_clicked:
    try:
        def convert_csv_to_jsonl(csv_file, jsonl_file):
            with open(csv_file, "r", encoding="utf-8") as csvfile, open(jsonl_file, "w", encoding="utf-8") as jsonlfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    json_line = json.dumps({
                        "prompt": row["prompt"],
                        "completion": row["completion"]
                    })
                    jsonlfile.write(json_line + "\n")

        convert_csv_to_jsonl(qa_csv_file, qa_jsonl_file)
        main_placefolder.text(f"Converted {qa_csv_file} to {qa_jsonl_file}")

        # Display fine-tuning command
        st.subheader("Fine-tune Command")
        st.code(f"openai api fine_tunes.create -t \"{qa_jsonl_file}\" -m \"gpt-3.5-turbo\"")

        st.success("Prepare fine-tuning using the command above!")
    except Exception as e:
        main_placefolder.error(f"Error preparing for fine-tuning: {str(e)}")

# Step 4: Query the Fine-tuned Model
query = st.text_input("Ask a question to the fine-tuned model:")
if query:
    if os.path.exists(index_file_path) and os.path.exists("docstore.pkl") and os.path.exists("index_to_docstore_id.pkl"):
        try:
            # Load FAISS index and details
            index = faiss.read_index(index_file_path)
            with open("docstore.pkl", "rb") as f:
                docstore = pickle.load(f)
            with open("index_to_docstore_id.pkl", "rb") as f:
                index_to_docstore_id = pickle.load(f)

            # Reconstruct vectorstore
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id=index_to_docstore_id)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display result
            st.header("Answer")
            st.write(result["answer"])

            # Display sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                st.write(sources)
        except Exception as e:
            st.error(f"Error querying the model: {str(e)}")