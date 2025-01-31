# pdf_processing.py


import pdfplumber
import camelot
import pandas as pd
from logging_config import logger

# Page extraction logic
def extract_page_no(pdf_path):
    """Extract the page number of the Profit & Loss statement."""
    try:
        relevant_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and "Statement of Profit and Loss" in text and "Revenue" in text and "Expenses" in text:
                    relevant_pages.append(i)
                    if len(relevant_pages) == 2:
                        return str(relevant_pages[1])
        return None
    except Exception as e:
        logger.error(f"Error extracting page number from PDF: {e}")
        return None

# Table extraction and preprocessing
def extract_profit_loss_tables(pdf_path, page):
    """Extract Profit & Loss tables from the PDF."""
    try:
        tables = camelot.read_pdf(pdf_path, pages=page, flavor='stream')
        if tables:
            df = tables[0].df
            df = df.iloc[2:].reset_index(drop=True)
            df.columns = df.iloc[0]
            df = df.drop(index=[0, 1])
            df[['Year ended March 31,2024', 'Year ended March 31,2023']] = df['Year ended March 31,'].str.split('\n', expand=True)
            df = df.drop(columns=['Note No.', 'Year ended March 31,'])
            df.columns.values[1] = 'Three months ended March 31,2024'
            df.columns.values[2] = 'Three months ended March 31,2023'
            df = df.reset_index(drop=True)
            return df
        else:
            logger.warning("No tables found on the specified page.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {e}")
        return pd.DataFrame()
