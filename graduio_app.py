import gradio as gr
import pandas as pd
import os
from rag import get_rag_pipeline

def create_master_df(amazon_file, cste_fk_file, gl_fk_file, meesho_file):
    amazon = pd.read_csv(amazon_file.name)
    cste_fk = pd.read_csv(cste_fk_file.name)
    gl_fk = pd.read_csv(gl_fk_file.name)
    meesho = pd.read_csv(meesho_file.name)

    amazon_subset = amazon[["Date", "Title", "MSKU", "Quantity"]]
    cste_fk_subset = cste_fk[["Dispatch by date", "Product", "SKU", "Quantity"]]
    gl_fk_subset = gl_fk[["Invoice Date (mm/dd/yy)", "Product", "SKU", "Quantity"]]
    meesho_subset = meesho[["Order Date", "Product Name", "SKU", "Quantity"]]

    amazon_subset["Date"] = pd.to_datetime(amazon_subset["Date"]).dt.date
    cste_fk_subset["Date"] = pd.to_datetime(cste_fk_subset["Dispatch by date"]).dt.date
    gl_fk_subset["Date"] = pd.to_datetime(gl_fk_subset["Invoice Date (mm/dd/yy)"]).dt.date
    meesho_subset["Date"] = pd.to_datetime(meesho_subset["Order Date"]).dt.date

    cste_fk_subset["Title"] = cste_fk_subset["Product"]
    cste_fk_subset["MSKU"] = cste_fk_subset["SKU"]
    gl_fk_subset["Title"] = gl_fk_subset["Product"]
    gl_fk_subset["MSKU"] = gl_fk_subset["SKU"]
    meesho_subset["Title"] = meesho_subset["Product Name"]
    meesho_subset["MSKU"] = meesho_subset["SKU"]

    amazon_subset["Quantity"] *= -1

    master_df = pd.concat([
        amazon_subset[["Date", "Title", "MSKU", "Quantity"]],
        cste_fk_subset[["Date", "Title", "MSKU", "Quantity"]],
        gl_fk_subset[["Date", "Title", "MSKU", "Quantity"]],
        meesho_subset[["Date", "Title", "MSKU", "Quantity"]],
    ], ignore_index=True)

    master_df = master_df.groupby(["Date", "Title", "MSKU"], as_index=False)["Quantity"].sum()
    master_df.to_csv("master_df.csv", index=False)
    return master_df, "master_df.csv"

rag_chat_fn = None  # Will be set after button click

with gr.Blocks() as app:
    gr.Markdown("### Upload Sales Data")

    with gr.Row():
        amazon_input = gr.File(label="Amazon", file_types=['.csv'])
        cste_fk_input = gr.File(label="CSTE FK", file_types=['.csv'])
        gl_fk_input = gr.File(label="GL FK", file_types=['.csv'])
        meesho_input = gr.File(label="Meesho", file_types=['.csv'])

    generate_button = gr.Button("Generate Master Data")
    output_df = gr.Dataframe()
    download_file = gr.File()

    generate_button.click(
        create_master_df,
        inputs=[amazon_input, cste_fk_input, gl_fk_input, meesho_input],
        outputs=[output_df, download_file]
    )

    gr.Markdown("### Chat with Your Sales Data")

    chatbot = gr.Chatbot(label="Sales RAG Bot", height=300)
    msg = gr.Textbox(label="Enter your question")
    chat_status = gr.Textbox(label="Status")

    def handle_chat_launch():
        global rag_chat_fn
        if not os.path.exists("master_df.csv"):
            return None, "❌ Please generate master_df.csv first"
        rag_chat_fn = get_rag_pipeline()
        return None, "✅ Chatbot is ready! Ask a question."

    launch_btn = gr.Button("Launch Chatbot")
    launch_btn.click(handle_chat_launch, outputs=[chatbot, chat_status])

    def run_chatbot(message, history):
        if not rag_chat_fn:
            return history + [(message, "❌ Chatbot not initialized")], ""
        response = rag_chat_fn(message)
        return history + [(message, response)], ""

    msg.submit(run_chatbot, inputs=[msg, chatbot], outputs=[chatbot, msg])

app.launch()
