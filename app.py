# app.py: This file will tie everything together, including the Gradio interface.

import gradio as gr
from pdf_processing import extract_page_no, extract_profit_loss_tables
from qa_processing import load_llm_models, prepare_context, answer_question_batch
from chroma_db import initialize_chroma_client, embed_and_store, load_embedding_model
from config import EXAMPLES
from logging_config import logger
from concurrent.futures import ThreadPoolExecutor

# Initialize components
initialize_chroma_client()
embedding_model = load_embedding_model()
llm_tokenizer, llm_model = load_llm_models("deepseek-ai/deepseek-coder-1.3b-instruct", "cpu")

# Function to handle PDF processing and queries
def process_pdf_and_queries(pdf_file, dropdown_queries, custom_queries):
    """Main function to handle PDF processing and query answering."""
    try:
        queries = custom_queries.split(",") if custom_queries else dropdown_queries
        
        pl_page = extract_page_no(pdf_file)

        with ThreadPoolExecutor() as executor:
            pnl_table_future = executor.submit(extract_profit_loss_tables, pdf_file, pl_page)
            pnl_table = pnl_table_future.result()

            embedding_future = executor.submit(embed_and_store, pnl_table, embedding_model, collection)
            embedding_future.result()  # Ensure embeddings are stored

        # Retrieve relevant rows for all queries in parallel
        relevant_rows = retrieve_relevant_rows_batch(queries, pnl_table)

        contexts = [prepare_context(relevant_rows)] * len(queries)
        answers = answer_question_batch(queries, contexts, llm_tokenizer, llm_model, "cpu")

        return relevant_rows, "\n".join(answers)
    except Exception as e:
        logger.error(f"Error processing PDF and queries: {e}")
        return None, "Error processing the queries."

# Gradio Interface setup
def build_gradio_interface():
    """Build the Gradio interface for the application."""
    example_pdf_path = "Sample Financial Statement.pdf"
    return gr.Interface(
        fn=process_pdf_and_queries,
        inputs=[
            gr.File(label="Upload PDF (P&L Statement)"),
            gr.Dropdown(
                label="Select multiple sample queries",
                choices=[
                    "What is the gross profit for Q3 2024?",
                    "What is the net income for 2024?",
                    "How much was the operating income for Q2 2024?",
                    "Show the operating margin for the past 6 months.",
                    "What are the total expenses for Q2 2023?"
                ],
                type="value",
                multiselect=True
            ),
            gr.Textbox(
                label="Or type custom queries (separate by comma) within quotes",
                placeholder="e.g., 'What is the net income for Q4 2024?, What is the operating margin for Q3 2024?'",
                lines=1,
                interactive=True
            )
        ],
        outputs=[
            gr.Dataframe(label="Retrieved financial data segments separated by ---", type="pandas", interactive=False),
            gr.Textbox(label="Answers", lines=10, interactive=False)
        ],
        title="Interactive Financial Data QA Bot",
        description="Upload a PDF with a P&L table and ask financial queries.",
        allow_flagging="never",
        examples=EXAMPLES,
    )

if __name__ == "__main__":
    iface = build_gradio_interface()
    iface.launch()
