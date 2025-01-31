# qa_processing.py: This file will handle context preparation, question answering, and retrieval.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from logging_config import logger
from torch import autocast
from config import LLM_MODEL_NAME, DEVICE

# Load LLM models
def load_llm_models():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    return tokenizer, model

# Context preparation
def prepare_context(relevant_rows):
    """Format the retrieved rows into a readable context."""
    try:
        if relevant_rows.empty:
            return ""
        formatted_data = []
        for index, row in relevant_rows.iterrows():
            row_str = ", ".join(f"{col}: {value}" for col, value in row.items())
            formatted_data.append(f"Row {index}: {row_str}")
        return "\n".join(formatted_data)
    except Exception as e:
        logger.error(f"Error preparing context: {e}")
        return ""

# Answer generation using LLM
def answer_question_batch(queries, contexts, tokenizer, model, device):
    """Generate answers for a batch of queries."""
    try:
        answers = []
        for query, context in zip(queries, contexts):
            if not context:
                answers.append("No relevant context found.")
                continue
            prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            inputs['input_ids'] = inputs['input_ids'].to(dtype=torch.long)
            with torch.no_grad():
                with autocast(device):
                    outputs = model.generate(
                        **inputs, max_new_tokens=200, temperature=0.01, top_p=0.9, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
            answers.append(answer)
        return answers
    except Exception as e:
        logger.error(f"Error generating answers for batch: {e}")
        return ["Error generating answer."] * len(queries)
