import streamlit as st
import psutil
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from docx import Document
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy
import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent

# Import LLMs and Embeddings
from langchain.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Word documents
def extract_text_from_word(word_files):
    text = ""
    for word in word_files:
        doc = Document(word)
        for paragraph in doc.paragraphs:
            text += paragraph.text
    return text

# Function to split extracted text into chunks
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# Function to create vector store with text embeddings
def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up conversational chain
def setup_conversational_chain(llm):
    template = """
    Based on the context, answer the question with as much detail as possible.
    If the answer is not available in the context, say "Answer not found in the provided context." 
    Do not generate an incorrect answer.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


# Main function to handle user question, response generation, and RAGAS evaluation
def handle_user_question(question, embeddings, llm):
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    documents = vector_store.similarity_search(question)

    retrieved_context = [doc.page_content for doc in documents]

    chain = setup_conversational_chain(llm)
    response = chain({"input_documents": documents, "question": question}, return_only_outputs=True)
    generated_answer = response["output_text"]
    
    st.write("Response:", generated_answer)

    evaluation_data = [{
        "question": question,
        "response": generated_answer,
        "retrieved_contexts": retrieved_context
    }]
    
    dataset = Dataset.from_list(evaluation_data)

    metrics = [faithfulness, answer_relevancy]

    results = evaluate(dataset, metrics)

    st.write("General Evaluation Results:")
    st.write(f"{results}")

# Function to process CSV file and return agent
def process_csv_file(csv_file, llm):
    if csv_file:
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_file_path = os.path.join(temp_dir, csv_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(csv_file.getbuffer())

        # Create the CSV agent using the LLM and the path to the CSV file
        agent = create_csv_agent(llm, temp_file_path, verbose=True, allow_dangerous_code=True,handle_parsing_errors=True)
        return agent
    return None

# Function to dynamically select the model and embeddings
def select_llm_model(choice):
    if choice == "Google Gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif choice == "Llama3.2":
        llm = Ollama(model="llama3.2")
        embeddings = OllamaEmbeddings(model="llama3.2")
    return llm, embeddings
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Multi-LLM File Chat")
    st.title("Multi-LLM File Chat 💡")

    st.sidebar.header("Select LLM")
    llm_choice = st.sidebar.selectbox("Choose a model to use:", ["Google Gemini", "Llama3.2"])
    llm, embeddings = select_llm_model(llm_choice)

    menu = ["Chat with PDF", "Chat with Word", "Chat with CSV"]
    choice = st.selectbox("Select File Type to Chat With", menu)

    if choice == "Chat with PDF":
        with st.sidebar:
            st.header("Upload PDFs")
            pdf_files = st.file_uploader("Select PDF files", accept_multiple_files=True, type="pdf")

            if st.button("Process PDFs"):
                if pdf_files:
                    st.info("Processing your PDFs...")
                    extracted_text = extract_text_from_pdfs(pdf_files)
                    chunks = split_text_into_chunks(extracted_text)
                    create_vector_store(chunks, embeddings)
                    st.success("PDFs processed successfully!")

    elif choice == "Chat with Word":
        with st.sidebar:
            st.header("Upload Word Documents")
            word_files = st.file_uploader("Select Word files", accept_multiple_files=True, type="docx")

            if st.button("Process Word Docs"):
                if word_files:
                    st.info("Processing your Word documents...")
                    extracted_text = extract_text_from_word(word_files)
                    chunks = split_text_into_chunks(extracted_text)
                    create_vector_store(chunks, embeddings)
                    st.success("Word documents processed successfully!")

    elif choice == "Chat with CSV":
        with st.sidebar:
            st.header("Upload CSV File")
            csv_file = st.file_uploader("Select CSV file", type="csv")

            if st.button("Process CSV"):
                if csv_file:
                    st.info("Processing your CSV file...")
                    agent = process_csv_file(csv_file, llm)
                    if agent:
                        st.success("CSV processed successfully!")
                    else:
                        st.error("Failed to process the CSV file.")

    user_question = st.text_input("Enter your question here:")

    if user_question:
        if choice == "Chat with CSV" and csv_file:
            agent = process_csv_file(csv_file, llm)
            if agent:
                    
                response = agent.run(user_question)
                start_time = time.time()
                response = agent.run(user_question)
                end_time = time.time()
                response_time = end_time - start_time

                memory_used = memory_usage()
                
                st.write("Response:", response)
                st.write("Response Time:", f"{response_time:.2f} seconds")
                st.write("Memory Usage:", f"{memory_used:.2f} MB")
                evaluation_data = [{
                "question": user_question,
                "response": response}]

                dataset = Dataset.from_list(evaluation_data)

                metrics = [answer_relevancy]
                results = evaluate(dataset, metrics)
                st.write("Evaluation Results:")
                st.write(f"{results}")
        else:
            handle_user_question(user_question, embeddings, llm)

if __name__ == "__main__":
    main()
