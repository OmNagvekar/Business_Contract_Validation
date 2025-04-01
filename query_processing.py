import logging
import fitz  # PyMuPDF
import pandas as pd
import torch
from tqdm import tqdm
from io import BytesIO

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core import Document

REQUESTS = 5
PERIOD = 60  # seconds



# Set up logging
logger = logging.getLogger(__name__)

# Set global embedding model settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en",
    cache_folder='../NanoScience/embed_model;'
)

# Define a refined project description and a chat prompt template.
PROJECT_DESCRIPTION = (
    "This project automates business contract validation by ensuring that contracts meet predefined standards. "
    "The system classifies input contracts and compares them against standard templates (including key provisions "
    "from the Indian Contract Law and the Indian Contract Act) to detect any deviations. "
    "Each clause must be evaluated for compliance with these legal standards. "
)

EXAMPLE_EVALUATION = (
    "For example, consider the clause: 'The contractor shall maintain confidentiality for 3 years after termination.' "
    "If the standard requires a confidentiality period of at least 2 years, then the correct evaluation would be 'Yes', "
    "with a rationale such as: 'The clause meets the minimum confidentiality period required by the standards.'"
)

CHAT_PROMPT_TEMPLATE = ChatPromptTemplate(
    [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Project Description: " + PROJECT_DESCRIPTION
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Example Evaluation: " + EXAMPLE_EVALUATION
        ),
        ChatMessage(
            role= MessageRole.SYSTEM,
            content= "History (for context only): {history}"
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Now, please evaluate whether the following contract clause adheres to the above standards. "
                "Respond with 'Yes' or 'No' and provide a brief rationale.\nClause: {clause}"
            )
        )
    ]
)


class QueryProcessor:
    def __init__(self, pdf_bytes: bytes,remote_llm:bool=False) -> None:
        logger.info("Initializing QueryProcessor")
        # Initialize the cancellation event if not already done.
        self.truth_values = None 
        self.chunks = None
        self.results = None 
        self.stop =False
        
        self.pdf_bytes = pdf_bytes # store the PDF in memory
        logger.info("PDF content loaded in memory (size: %d bytes)", len(pdf_bytes))
        
        # Initialize the vector index creator
        self.vector_exits = CreateVectorIndex('./').create_indexes()
        
        parser = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". "],
            chunk_size=1000,       # adjust chunk size as needed
            chunk_overlap=200      # adjust overlap to retain context between chunks
        )
        self.text_splitter = LangchainNodeParser(lc_splitter=parser)

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
            self.llm = Gemini(
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
            doc_classification = ds.DocumentClassification(pdf_bytes=self.pdf_bytes,model=self.model)
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
        logger.info("Opening PDF from in-memory bytes")
        try:
            pdf_stream = BytesIO(self.pdf_bytes)
            doc = fitz.open(stream=pdf_stream,filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF file: {e}")
            return []
        
        text = ""
        for page in doc:
            text += page.get_text()
            
        documents = [Document(text=text)]
        nodes = self.text_splitter.get_nodes_from_documents(documents)
        chunks = [node.get_content() for node in nodes]
        logger.info(f"Extracted text and split into {len(chunks)} chunks")
        return chunks
    
    @sleep_and_retry
    @limits(calls=REQUESTS, period=PERIOD)
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=1, max=30))
    def query_clause(self, history:str,clause: str):
        
        if self.stop:
            logger.info("Cancellation detected before API call. Returning 'Cancelled' response.")
            return clause, "Cancelled", False
        
        response = None
        try:
            logger.info("Querying clause through LLM")
            query_prompt = CHAT_PROMPT_TEMPLATE.format(history=history,clause=clause)
            response = self.query_engine.query(query_prompt)
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
        self.clauses =[]
        logger.info(f"Processing {len(self.chunks)} chunks for alignment")
        for count,chunk in enumerate(tqdm(self.chunks, desc="Evaluating chunks"),start=1):
            if self.stop:
                logger.info("Cancellation requested. Halting alignment checking.")
                break

            # Build history from the previous clause and response if available.
            if self.results and self.clauses:
                history = f"Previous Clause: {self.clauses[-1]}\nPrevious Response: {self.results[-1]}"
            else:
                history = ""

            clause, result, truth_value = self.query_clause(history,chunk)
            if result == "Cancelled":
                logger.info("Cancellation detected. Stopping processing.")
                break
            self.results.append(result)
            self.truth_values.append(truth_value)
            self.clauses.append(clause)
            count += 1
            logger.debug(f"Processed chunk {count}: Truth value = {truth_value}")
            logger.info(f"\nclause {clause}\n, result {result}\n, truth_value {truth_value}\n")
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
        needles = [clause for clause, truth in zip(df['Clause'], df['Truth_Value']) if not truth]
        logger.info(f"Found {len(needles)} clauses that do not align with the documents")
        
        try:
            pdf_stream = BytesIO(self.pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF for highlighting: {e}")
            return
        
        for needle in tqdm(needles, desc="Highlighting needles"):
            if self.stop:
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
        try:
            pdf_buffer = BytesIO()
            doc.save(pdf_buffer)
            pdf_buffer.seek(0)
            pdf= pdf_buffer.read()
            logger.info(f"Highlighted PDF saved")
            return pdf,df.to_csv(index=False).encode("utf-8")
        except Exception as e:
            logger.error(f"Failed to save highlighted PDF: {e}")

if __name__ == "__main__":
    logger.info("Starting QueryProcessor main execution")
    qp = QueryProcessor(pdf_bytes=bytes(open("./exhibit101.pdf","rb").read()),remote_llm=True)
    qp.checking_alignment()
    qp.pdf_highlighter()
    logger.info("Execution finished")
