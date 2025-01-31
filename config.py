# config.py:This file will store constants and configuration details that are shared across different modules.

import torch

# Constants
DEVICE = "cpu"  # Default fallback to CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
CHROMA_DB_PATH = "./chroma_db"
EXAMPLES = [["Sample Financial Statement.pdf"]]
