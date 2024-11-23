# Chatbot Application

https://github.com/user-attachments/assets/0793644b-3630-4d67-9479-8fe3b3a71aa4


This chatbot application is built using Python and various libraries, including Streamlit, LangChain, and LangChain Community. It allows users to upload PDF files or provide URLs of PDFs, and then interacts with the extracted text using a language model.

## Features

1. Upload multiple PDF files or provide URLs of PDFs.

2. Extract text from the uploaded PDF files and URLs.

3. Store the extracted text in a vector store for efficient retrieval.
4. Perform a Wikipedia search based on the user's question.
5. Perform an Arxiv search based on the user's question.
6. Route the user query to the most relevant datasource (Wikipedia, Arxiv, or vector store).
7. Generate a response based on the retrieved documents from the selected datasource.
## Installation

1. Clone the repository:

2. Navigate to the project directory:

3. Install the required libraries:

4. Run the application:

5. Create a `.env` file in the project directory and add your Astra DB, Groq API, and other necessary environment variables:

6. Run the application:


## Usage

1. Open your web browser and navigate to `http://localhost:8501`.
2. Upload PDF files or provide URLs of PDFs.
3. Ask questions using the chatbot interface.
4. The chatbot will extract text from the uploaded PDFs, store it in the Cassandra database, and use the LLM to generate responses.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
