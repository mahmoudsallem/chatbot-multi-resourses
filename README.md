# Chatbot Application

This chatbot application is built using Python and various libraries, including Streamlit, LangChain, and LangChain Community. It allows users to upload PDF files or provide URLs of PDFs, and then interacts with the extracted text using a language model.

## Features

1. Upload multiple PDF files.
2. Provide URLs of PDFs.
3. Extract text from PDFs.
4. Store the extracted text in a Cassandra database.
5. Use a language model (LLM) to answer questions based on the stored text.

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
