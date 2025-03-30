import os
import time
import logging
import fitz  # PyMuPDF
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core import load_index_from_storage, StorageContext, Settings

from creating_indexes_and_storing import CreateVectorIndex
import tensorflow as tf
import keras
import document_classification as ds  # assuming this module is needed later
from llama_index.llms.gemini import Gemini
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

REQUESTS = 5
PERIOD = 60  # seconds

# Global cancellation event
STOP_EVENT = None


# Set up logging
logger = logging.getLogger(__name__)

# Set global embedding model settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en",
    cache_folder='../NanoScience/embed_model;'
)

# Define a refined project description and a chat prompt template.
PROJECT_DESCRIPTION = (
    "This project aims to automate business contract validation by ensuring that contracts meet "
    "predefined standards. The system first classifies the input contracts and then compares them "
    "against standard templates to detect any deviations or potential issues."
)

CHAT_PROMPT_TEMPLATE = ChatPromptTemplate(
    [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Project Description: " + PROJECT_DESCRIPTION
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Please evaluate whether the following contract clause adheres to the above standards. "
                "Respond with 'Yes' or 'No' and provide a brief rationale.\nClause: {clause}"
            )
        )
    ]
)

class QueryProcessor:
    def __init__(self, input_pdf: str,remote_llm:bool=False) -> None:
        logger.info("Initializing QueryProcessor")
        global STOP_EVENT
        # Initialize the cancellation event if not already done.
        if STOP_EVENT is None:
            from threading import Event
            STOP_EVENT = Event()

        self.truth_values = None 
        self.chunks = None
        self.results = None 
        self.stop =False
        
        self.input_pdf = input_pdf
        logger.info(f"Input PDF: {self.input_pdf}")
        
        # Initialize the vector index creator
        self.vector_exits = CreateVectorIndex('./')
        
        # Set up text splitter
        self.text_splitter = TokenTextSplitter(
            separator=" ", 
            chunk_size=50,
            chunk_overlap=20
        )
        
        # Load the classification model
        try:
            logger.info("Loading TensorFlow model for document classification")
            self.model = tf.keras.models.load_model("./Document_classification3.keras")
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            self.model = None
            raise e
        
        # Set up device and initialize LLM (Ollama)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device} for LLM")
        if remote_llm:
            with open("gemini_key.txt",'r') as f:
                key = f.read()
            self.llm = llm = Gemini(
                model="models/gemini-2.0-flash",
                api_key=key,  # uses GOOGLE_API_KEY env var by default
                temperature=0.5,
            )
            logger.info(f"Using Remote LLM: gemini-2.0-flash")
        else:
            self.llm = Ollama(model="phi3:mini", request_timeout=360.0, device=device)
            logger.info(f"Using Local LLM: phi3:mini")
        
        # Load vector index from storage
        try:
            # For demonstration, the directory is hard-coded. Consider dynamic selection.
            doc_classification = ds.DocumentClassification(path=self.input_pdf,model=self.model)
            persist_dir = f"./vector_indexes/{doc_classification.classify_doc()}/"
            # persist_dir = "./vector_indexes/Employment Agreements/"
            logger.info(f"Loading vector index from: {persist_dir}")
            self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))
            self.query_engine = self.index.as_query_engine(llm=self.llm)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = None
            self.query_engine = None

    def pdf_to_chunks(self):
        logger.info(f"Opening PDF file: {self.input_pdf}")
        try:
            doc = fitz.open(self.input_pdf)
        except Exception as e:
            logger.error(f"Failed to open PDF file: {e}")
            return []
        
        text = ""
        for page in doc:
            text += page.get_text()
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Extracted text and split into {len(chunks)} chunks")
        return chunks
    
    @sleep_and_retry
    @limits(calls=REQUESTS, period=PERIOD)
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=1, max=30))
    def query_clause(self, clause: str):
        
        if STOP_EVENT.is_set() or self.stop:
            logger.info("Cancellation detected before API call. Returning 'Cancelled' response.")
            return clause, "Cancelled", False
        
        response = None
        try:
            logger.info("Querying clause through LLM")
            query_prompt = CHAT_PROMPT_TEMPLATE.format(clause=clause)
            query_text = f"Evaluate whether the following clause aligns with the given documents. Provide an answer in yes or no: {clause}"
            response = self.query_engine.query(query_text)
            logger.info(f"Response of LLM: {str(response)}")
        except Exception as e:
            logger.error(f"Query failed: {e}. Defaulting response to 'yes'")
            response = "yes"
        result = str(response)
        truth_value = "yes" in result.lower()
        return clause, result, truth_value

    def checking_alignment(self):
        logger.info("Starting document alignment check")
        self.chunks = self.pdf_to_chunks()
        if not self.chunks:
            logger.error("No chunks available for processing.")
            return
        
        self.results = []
        self.truth_values = []
        count = 0
        logger.info(f"Processing {len(self.chunks)} chunks for alignment")
        for count,chunk in enumerate(tqdm(self.chunks, desc="Evaluating chunks"),start=1):
            if STOP_EVENT.is_set() or self.stop:
                logger.info("Cancellation requested. Halting alignment checking.")
                break
            clause, result, truth_value = self.query_clause(chunk)
            self.results.append(result)
            self.truth_values.append(truth_value)
            count += 1
            logger.debug(f"Processed chunk {count}: Truth value = {truth_value}")
        logger.info("Completed alignment checking.")

    def gen_df(self):
        logger.info("Generating dataframe from results")
        return pd.DataFrame({
            "Clause": self.chunks, 
            "Truth_Value": self.truth_values,
            "Responses": self.results
        })

    def pdf_highlighter(self):
        logger.info("Starting PDF highlighting process")
        df = self.gen_df()
        df.to_csv("comment.csv", index=False)
        needles = [clause for clause, truth in zip(df['Clause'], df['Truth_Value']) if not truth]
        logger.info(f"Found {len(needles)} clauses that do not align with the documents")
        
        try:
            doc = fitz.open(self.input_pdf)
        except Exception as e:
            logger.error(f"Failed to open PDF for highlighting: {e}")
            return
        
        for needle in tqdm(needles, desc="Highlighting needles"):
            if STOP_EVENT.is_set() or self.stop:
                logger.info("Cancellation requested. Halting PDF highlighting.")
                break
            for page_num, page in enumerate(doc, start=1):
                rects = page.search_for(needle)
                if rects:
                    logger.info(f"Highlighting found on page {page_num} for needle: {needle[:30]}...")
                    # Highlight the area from first to last rectangle found on the page
                    p1 = rects[0].tl
                    p2 = rects[-1].br
                    page.add_highlight_annot(start=p1, stop=p2)
        output_pdf = "new.pdf"
        try:
            doc.save(output_pdf)
            logger.info(f"Highlighted PDF saved as {output_pdf}")
        except Exception as e:
            logger.error(f"Failed to save highlighted PDF: {e}")

if __name__ == "__main__":
    logger.info("Starting QueryProcessor main execution")
    qp = QueryProcessor(input_pdf="./exhibit101.pdf")
    qp.checking_alignment()
    qp.pdf_highlighter()
    logger.info("Execution finished")
