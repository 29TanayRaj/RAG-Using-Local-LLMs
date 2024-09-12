## RAG+LLM Web App
This web app integrates Retrieval-Augmented Generation (RAG) with local large language models (LLMs) to provide accurate, context-aware responses based on PDF documents. The app is built using Streamlit for the front end, Ollama for local LLMs, and FAISS for efficient information retrieval.

Key Features:
PDF Upload & Processing: Users can upload PDF files, and the app extracts their content for processing.
Retrieval-Augmented Generation (RAG): The system retrieves relevant information from stored PDFs using FAISS before passing it to the LLM for generating responses, ensuring fact-based answers.
Local LLMs (Ollama): The app uses local LLMs to generate human-like responses, keeping all operations local for enhanced privacy and performance.
Efficient Retrieval with FAISS: The FAISS database stores embeddings of the PDF content, enabling fast and accurate search results.
Chat History: The app stores chat history, allowing users to maintain context across conversations.
This tool is ideal for anyone looking to build an interactive document-based assistant that leverages the power of local language models combined with efficient retrieval systems.
