# 🧠 Sales Data RAG Chatbot

A **Gradio-based web app** to upload sales data from multiple marketplaces, generate a unified master dataset, and ask questions using a **Retrieval-Augmented Generation (RAG)** chatbot powered by Hugging Face.

---

## 📦 What I Built

### 🔧 Tech Stack
- **Frontend/UI**: `Gradio`
- **Backend**: 
  - `pandas` for CSV processing
  - `langchain` for RAG logic
  - `FAISS` for vector search
  - `HuggingFace Embeddings` and `LLMs` (Phi-3)

### 🧠 Logic
1. Accept CSVs from **Amazon, CSTE FK, GL FK**, and **Meesho**
2. Preprocess & merge them into a `master_df.csv`
3. Convert each row into natural-language-style documents
4. Use FAISS to perform semantic search on the documents
5. Use a Hugging Face-hosted model (`microsoft/Phi-3-mini-4k-instruct`) to answer user queries

---

## 🏗️ How I Built It

### 🛠️ AI Tools Used
- `HuggingFaceEndpoint`: Calls an open-source LLM via Hugging Face Inference API
- `HuggingFaceEmbeddings`: Generates dense embeddings from tabular data
- `FAISS`: Performs fast similarity search to retrieve relevant rows
- `LangChain`: Combines the retriever, embeddings, and LLM into a RAG pipeline

---

## 🚀 How to Use

1. Upload `.csv` files from each platform:
   - **Amazon**
   - **CSTE FK**
   - **GL FK**
   - **Meesho**
2. Click **"Generate Master Data"** to create `master_df.csv`
3. Click **"Launch Chatbot"** to initialize the RAG pipeline
4. Enter natural language queries like:
   - `"How many SKUs were sold on May 3rd?"`
   - `"Total quantity of MSKU-123 across all channels?"`

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sales-rag-chatbot.git
cd sales-rag-chatbot
```

### 2. Install Dependencies
Make sure you have Python 3.8+ and run:
```bash
pip install -r requirements.txt
```

### 3. Add Hugging Face API Key

You **must** set your Hugging Face API token to use the chatbot.

Option 1: Set environment variable
```bash
export HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

Option 2: Login via CLI
```bash
huggingface-cli login
```

> ⚠️ Without a Hugging Face API token, the chatbot will not function.

---

## 📁 Folder Structure

```
.
├── gradio_app.py        # Gradio UI and orchestration
├── rag.py               # RAG logic (embeddings, retriever, LLM)
├── README.md
├── requirements.txt     # Python dependencies
```

---

## 💡 Example Queries

- `"Show all sales on April 5th"`
- `"Quantity sold for MSKU-456"`
- `"Which products had negative quantity?"`

---

## 📌 Notes

- Only `.csv` files are accepted for input.
- `master_df.csv` must be generated before chatbot interaction.
- Tested using `microsoft/Phi-3-mini-4k-instruct` model from Hugging Face.

---

## 🧾 Requirements

Below are the required Python packages:

```
gradio
pandas
langchain
faiss-cpu
langchain-community
langchain-huggingface
huggingface_hub
```

---

## 📬 Contact

Built with ❤️ by **Your Name**

Feel free to open an issue or contribute via pull request.
