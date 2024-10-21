# Chat with Multiple PDFs

## Introduction

The **MultiPDF Chat App** allows users to interact with multiple PDF documents through a chatbot interface. By leveraging **LangChain** and **OpenAI**, the app extracts content from the PDFs, converts it into semantic embeddings, and allows users to ask questions in natural language. The app retrieves the most relevant information from the PDFs and generates responses based on the user's queries.

## Architecture
![LangChain PDF Processing Architecture](docs/app_architecture.png)

The architecture of the MultiPDF Chat App consists of several stages:

1. **PDF Input**: Users can upload multiple PDF files to the application.
2. **Text Chunking**: The PDFs are processed and broken down into smaller chunks of text for efficient embedding generation.
3. **Embedding Creation**: The app uses **OpenAI** to generate semantic embeddings for each chunk of text.
4. **Vector Store**: Embeddings are stored in a vector database for easy retrieval.
5. **Question Processing**: When the user submits a question, it is transformed into a semantic embedding.
6. **Semantic Search**: The app searches the vector store to find the most relevant chunks of text.
7. **Answer Generation**: The app sends the most relevant chunks to a **language model** to generate a response.

## Dependencies and Installation

To install the MultiPDF Chat App, please follow these steps:

1. **Clone the repository** to your local machine:
   ```bash
   https://github.com/elmontaser1998/Chat_with_multiple_pdfs.git
   cd chat-with-multiple-pdfs
   
2. Install the required dependencies by running the following command:
     ```bash
     pip install -r requirements.txt
