# RAG+LLM chatbot app 📚🔍🧠

This web app integrates **Retrieval-Augmented Generation (RAG)** with local **Large Language Models (LLMs)**, enabling accurate, context-aware responses from PDF documents. The app uses **Streamlit** for the front end, **Ollama** for local LLMs, and **FAISS** for fast, efficient retrieval of information.

<img src="https://github.com/user-attachments/assets/7c33221a-2d15-45d9-aaf2-ec5fe384e478" alt="rag" width="600"/>


### Key Features

- **PDF Upload & Processing**: Users can upload PDFs, and the app will extract their content for processing.
- **Retrieval-Augmented Generation (RAG)**: Information from the uploaded PDFs is retrieved using **FAISS**, ensuring fact-based responses. Relevant sections of the documents are passed to the LLM to generate answers.
- **Local LLMs (Ollama)**: The app uses local models like **Llama 3.1**, **Mistral**, **Gemma 2**, and **LLaVA**, ensuring all computations and data stay local for enhanced privacy and performance.
- **Efficient Retrieval with FAISS**: The **FAISS** database stores embeddings of the PDF content using **nomic-embed-text**, enabling fast and accurate search results.
- **Chat History**: Users can maintain and restore chat history, preserving context across conversations.
  
### Tech Stack

- **Front End**: Built with **Streamlit** for an easy-to-use interface.
- **LLMs**: Uses local models like **Llama 3.1**, **Mistral**, **Gemma 2**, and **LLaVA** from **Ollama**.
- **Embeddings**: Generated using **nomic-embed-text**.
- **Document Retrieval**: Powered by **FAISS** for fast and efficient similarity search.

### How It Works

1. **Upload a PDF**: Users upload a PDF, which is processed into chunks for easy retrieval.
2. **Embedding Creation**: The content is split into chunks and stored in the FAISS database after generating embeddings.
3. **Ask a Question**: Users can ask questions, and the app retrieves relevant document sections before passing the context to the LLM for generating a response.
4. **Chat History**: Conversation history is stored, allowing users to continue discussions without losing context.

### Requirements

- **Streamlit** for the front end.
- **Ollama** for local LLM models.
- **FAISS** for embedding-based document retrieval.

### How to use this app in your local system.

### Usage
To get started, clone the repository and follow the instructions below to set up the environment and start the app.

- **Setup a Enviroment**: Use python 3.10.0 or above to setup a virtual environment in python, one can use virtualenv for the same.
- **Installing the Libraries**: Use the requirements.txt file to install the required python dependencies.
- **Ollama**: Install Ollama, and download models **Llama 3.1**, **Mistral**, **Gemma 2**, and **LLaVA** for LLMS and **nomic-embed-text** for embeddings. Note: One does not need to install all the models just install a single LLM -(like llama3.1) but do install the **nomic-embed-text** for embeddings. also make the necessary changes in the code.
- **Running the APP**: In the ternmial open the folder in which the file **app.py** is stored and type the command: streamlit run app.py
---
#### References: 
These resources helped me complete this project:

1. What is RAG? (Retrieval Augmented Generation) on Clarifai Computer Vision By Ian Kelk
https://www.clarifai.com/blog/what-is-rag-retrieval-augmented-generation

2. Krish Naik's Updated langchain Playlist:https://youtube.com/playlist?list=PLZoTAELRMXVOQPRG7VAuHL--y97opD5GQ&feature=shared

3. Alejandro AO - Software & Ai youtube channel : https://www.youtube.com/@alejandro_ao

