from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.llms.ollama import Ollama
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core import load_index_from_storage, StorageContext, Settings
import fitz  # PyMuPDF is imported as fitz
import document_classification as ds
import pandas as pd
import torch
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import logging
from tqdm import tqdm
from creating_indexes_and_storing import CreateVectorIndex
import keras
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO)

# os.environ["GRADIENT_ACCESS_TOKEN"] = "YOUR_GRADIENT_ACCESS_TOKEN_HERE"
# os.environ["GRADIENT_WORKSPACE_ID"] = "YOUR_GRADIENT_WORKSPACE_ID_HERE"
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en",cache_folder='../NanoScience/embed_model;')

class QueryProcessor:
    def __init__(self, input_pdf):
        self.truth_values = None 
        self.chunks = None
        self.results = None 
        self.vector_exits=CreateVectorIndex('./')
        self.input_pdf = input_pdf
        self.text_splitter = TokenTextSplitter(
            separator=" ", 
            chunk_size=50,
            chunk_overlap=20
        ) 
        print(type(self.input_pdf))
        print(self.input_pdf)
        model = tf.keras.models.load_model("./Document_classification3.keras")
        doc_classification = ds.DocumentClassification(path=self.input_pdf,model=model)
        print(doc_classification.classify_doc())
        # llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        # self.pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
        # llm = HuggingFaceLLM(
        #     context_window=4096,
        #     max_new_tokens=2000,
        #     device_map="auto",
        #     generate_kwargs={"temperature":0.0, "do_sample":False},
        #     model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        #     tokenizer_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        #     stopping_ids=[50278, 50279, 50277, 1, 0],
        #     tokenizer_kwargs={"max_length":4096},
        #     model_kwargs={"torch_dtype":torch.float16, "load_in_4bit":True}
        # ) 

        # llm = GradientBaseModelLLM(
        #     base_model_slug="llama2-7b-chat",
        #     max_tokens=510,
        # )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.llm = Ollama(model="llama3", request_timeout=360.0,device=device)

        self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=f"./vector_indexes/{doc_classification.classify_doc()}/"))
        self.query_engine = self.index.as_query_engine(llm=self.llm)

    def pdf_to_chunks(self):
        doc = fitz.open(self.input_pdf)
        text = ""
        for page in doc:
            text += page.get_text()
        return self.text_splitter.split_text(text)

    def query_clause(self, clause):        
        response = None
        try: 
            response = self.query_engine.query(f"""Evaluate whether the following clause aligns with the given documents. Provide an answer in yes or no: {clause}""")
        except:
            response = "yes"
        result = str(response)

        truth_value = "yes" in result.lower()
        return clause, result, truth_value

    def checking_alignment(self):
        self.chunks = self.pdf_to_chunks()
        self.results = []
        self.truth_values = []
        count = 0
        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.query_clause, clause) for clause in self.chunks]
        #     for future in tqdm(as_completed(futures)):
        #         clause, result, truth_value = future.result()
        #         self.results.append(result)
        #         self.truth_values.append(truth_value)
        #         if count == len(self.chunks):
        #            break

        #         else:
        #             count+=1

        print(len(self.chunks))
        for chunk in tqdm(self.chunks):
            clause,result,truth_value = self.query_clause(chunk)
            time.sleep(0.3)
            #print(f"result {result}")
            self.results.append(result)
            self.truth_values.append(truth_value) 
           
            #print(self.truth_values)
            #return None

    def gen_df(self):
        return pd.DataFrame({"Clause": self.chunks, "Truth_Value": self.truth_values,"Responses":self.results})

    def pdf_highlighter(self):
        df = self.gen_df()
        df.to_csv("comment.csv")
        #print(df)
        needles = [a for a, b in zip(df['Clause'], df['Truth_Value']) if not b]
        print(len(needles)) 
        doc = fitz.open(self.input_pdf)

        for needle in tqdm(needles):
            for page in doc:
                rects = page.search_for(needle)
                
                if len(rects)!= 0: 
                    p1 = rects[0].tl
                    p2 = rects[-1].br
                    page.add_highlight_annot(start=p1,stop=p2)                
                
                    
        doc.save("new.pdf")

if __name__ == "__main__":
    obj = QueryProcessor(input_pdf="./exhibit101.pdf")
    obj.checking_alignment()
    obj.pdf_highlighter()
