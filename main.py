import os
import streamlit as st
import pickle
import time
import csv
import random
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import faiss
from textblob import TextBlob
import openai
import json

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI Setup
st.title("End-to-End News Research Tool with Sentiment Analysis and Fine-tuning")
st.sidebar.title("Input URL")

# Input URL
url = st.sidebar.text_input("URL")

# Buttons for actions
process_url_clicked = st.sidebar.button("Process URL")
generate_pairs_clicked = st.sidebar.button("Generate Q&A Pairs")
fine_tune_clicked = st.sidebar.button("Fine-tune the Model")
index_file_path = "faiss_index"
qa_csv_file = "qa_dataset.csv"
scraped_data_file = "scraped_data.txt"
fine_tuning_file = "fine_tuning_data.jsonl"

main_placeholder = st.empty()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

# Step 1: Scrape Data from URL
if process_url_clicked:
    try:
        loader = UnstructuredURLLoader(urls=[url])
        main_placeholder.text("Data Loading ... Started")
        data = loader.load()

        # Save raw data to a text file
        with open(scraped_data_file, "w", encoding="utf-8") as f:
            for doc in data:
                f.write(doc.page_content + "\n\n")
        main_placeholder.text(f"Raw data saved to {scraped_data_file}")

        # Split data for embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )

        main_placeholder.text("Text Splitting ... Started")
        docs = text_splitter.split_documents(data)

        # Sentiment analysis
        def analyze_sentiment(text):
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            return sentiment

        sentiment_scores = [analyze_sentiment(doc.page_content) for doc in data]
        sentiment_avg = sum(sentiment_scores) / len(sentiment_scores)

        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building")
        time.sleep(2)

        # Save FAISS index and related details
        faiss.write_index(vectorstore_openai.index, index_file_path)
        with open("docstore.pkl", "wb") as f:
            pickle.dump(vectorstore_openai.docstore, f)
        with open("index_to_docstore_id.pkl", "wb") as f:
            pickle.dump(vectorstore_openai.index_to_docstore_id, f)

        main_placeholder.text("URL processed successfully!")

        # Display Sentiment Analysis Result
        st.subheader("Sentiment Analysis")
        st.write(f"Average Sentiment Score: {sentiment_avg:.2f}")
        if sentiment_avg > 0:
            st.write("The content is generally **positive**.")
        elif sentiment_avg < 0:
            st.write("The content is generally **negative**.")
        else:
            st.write("The content is **neutral**.")
        
    except Exception as e:
        main_placeholder.error(f"Error processing URL: {str(e)}")

# Step 2: Generate Q&A Pairs
if generate_pairs_clicked:
    try:
        def generate_pairs(input_file, output_csv):
            example_questions = [
                "What is the main topic of this paragraph?",
                "Explain the key insights mentioned here.",
                "Summarize this content briefly.",
                "What is the conclusion of this section?",
                "What are the major findings in this article?",
                "What does the author suggest as the next steps?",
                "What are the key arguments presented?",
                "How does this information relate to the overall theme?",
                "What evidence does the author provide to support their claims?",
                "Can you provide a brief summary of the article's introduction?"
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
        main_placeholder.text(f"Generated {count} Q&A pairs and saved to {qa_csv_file}")
    except Exception as e:
        main_placeholder.error(f"Error generating Q&A pairs: {str(e)}")

# Step 3: Fine-tuning the Model
if fine_tune_clicked:
    try:
        # Convert the CSV to JSONL format required for fine-tuning
        def generate_fine_tuning_data(input_file):
            training_data = []
            with open(input_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Prepare data in the correct format for fine-tuning
                    training_data.append({
                        "prompt": row["prompt"],
                        "completion": row["completion"]
                    })
            return training_data

        # Generate training data and save to a JSONL file
        training_data = generate_fine_tuning_data(qa_csv_file)
        with open(fine_tuning_file, "w", encoding="utf-8") as f:
            for entry in training_data:
                json.dump(entry, f)
                f.write("\n")

        # Upload the JSONL file for fine-tuning
        response = openai.File.create(
            file=open(fine_tuning_file, "r"),
            purpose="fine-tune"
        )

        file_id = response["id"]

        # Fine-tune the model
        fine_tune_response = openai.FineTune.create(
            training_file=file_id,
            model="gpt-3.5-turbo"  # Base model to fine-tune
        )

        fine_tune_model_id = fine_tune_response['fine_tuned_model']
        main_placeholder.text(f"Model fine-tuned successfully. Fine-tuned model ID: {fine_tune_model_id}")

    except Exception as e:
        main_placeholder.error(f"Error during fine-tuning: {str(e)}")

# Step 4: Query the Fine-tuned Model
question_options = [
    "What is the main topic of this paragraph?",
    "Explain the key insights mentioned here.",
    "Summarize this content briefly.",
    "What is the conclusion of this section?",
    "What are the major findings in this article?",
    "What does the author suggest as the next steps?",
    "What are the key arguments presented?",
    "How does this information relate to the overall theme?",
    "What evidence does the author provide to support their claims?",
    "Can you provide a brief summary of the article's introduction?"
]

query = st.selectbox("Select a question to ask the model:", question_options)

if query:
    if os.path.exists(index_file_path) and os.path.exists("docstore.pkl") and os.path.exists("index_to_docstore_id.pkl"):
        try:
            # Load FAISS index and details
            index = faiss.read_index(index_file_path)
            with open("docstore.pkl", "rb") as f:
                docstore = pickle.load(f)
            with open("index_to_docstore_id.pkl", "rb") as f:
                index_to_docstore_id = pickle.load(f)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)

            # Perform retrieval and answer generation
            retrieval_qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm,
                chain_type="stuff", 
                retriever=vectorstore.as_retriever()
            )
            
            # Correct input structure: pass 'question' key
            result = retrieval_qa.invoke({"question": query})  # Use 'question' instead of 'input'

            answer = result["answer"]
            sources = result["sources"]

            # Display the answer and sources
            main_placeholder.text(f"Answer: {answer}")
            st.subheader("Sources")
            st.write(sources)

        except Exception as e:
            main_placeholder.error(f"Error querying the model: {str(e)}")