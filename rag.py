import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import Document

def get_rag_pipeline():
    if not os.path.exists("master_df.csv"):
        raise FileNotFoundError("master_df.csv not found.")

    df = pd.read_csv("master_df.csv")

    documents = [
        Document(page_content=f"Date: {row['Date']}, Title: {row['Title']}, MSKU: {row['MSKU']}, Quantity: {row['Quantity']}")
        for _, row in df.iterrows()
    ]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    llm = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        max_new_tokens=256,
        do_sample=False,
    )
    chat = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        input_variables=["results", "query"],
        template="Given the data below:\n\n{results}\n\nAnswer the question: \"{query}\""
    )

    def rag_chat(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n".join(doc.page_content for doc in docs)
        prompt_text = prompt.format(results=context, query=query)
        result = chat.invoke([{"role": "user", "content": prompt_text}])
        return result.content

    return rag_chat

