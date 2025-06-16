# 🔍 AI-Powered GitHub Repository Explorer (RAG-Based)

A powerful tool to interact with any GitHub repository using Retrieval-Augmented Generation (RAG). This project allows you to fetch and analyze codebases directly from GitHub, ask intelligent questions, and receive accurate answers—powered by an LLM.

---

## 🚀 Features

- ✅ Fetch and process any GitHub repository
- ✅ Clean and chunk files for optimized analysis
- ✅ Store document embeddings using ChromaDB
- ✅ Retrieve relevant context dynamically with RAG
- ✅ Query the repo via command line and receive insightful answers
- ✅ Fully LLM-powered with support for custom API keys

---

## 📁 How It Works

1. **Fetch the Repository**  
   Uses `GithubLoader` to load all files from a given GitHub repository.

2. **Parse and Clean**  
   Files are parsed and cleaned to remove unnecessary white space and noise.

3. **Chunk the Data**  
   Cleaned data is divided into manageable chunks for efficient processing.

4. **Embed and Store**  
   Chunks are converted into vector embeddings and stored in **ChromaDB** (used for the RAG pipeline).

5. **Query and Retrieve**  
   When a question is asked, the most relevant chunks are retrieved from ChromaDB based on semantic similarity.

6. **Generate a Response**  
   Retrieved content is injected into a prompt and passed to a **LLM via LLMChain**, which generates a final response.

---

## 🛠️ Setup Instructions

- Clone the repository
- Install dependencies
- Add your **API keys** for the embedding model and LLM (e.g., OpenAI)
- Run the project via command line

---

## 💡 Tips for Best Results

- Be specific in your queries. Mention relevant filenames or components.
- Ensure that the repository is public or your credentials support access.
- Larger repositories may take more time to process initially.

---

## 📅 Coming Soon

- 🖥️ Interactive Web UI for easier usage
- 📦 Support for multiple vector stores
- 📊 Visual insights into code structure

---

## 🔐 API Keys

Make sure to provide your own API keys:
- LLM provider (e.g., OpenAI)
- Embedding model (e.g., OpenAI or HuggingFace)

---

**Author:** Sayantan Sarkar

---
