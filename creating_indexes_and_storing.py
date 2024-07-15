from llama_index.core import load_index_from_storage,StorageContext
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core  import VectorStoreIndex
from tqdm import tqdm

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")

path = "./"
files = os.listdir("./data")
for i in tqdm(files):
    doc = SimpleDirectoryReader(input_dir=f"{path}/data/{i}").load_data()
    os.system(f"mkdir {path}/new_data/{i}")
    index = VectorStoreIndex(doc,embed_model=embed_model)
    index.storage_context.persist(persist_dir = f"{path}/new_data/{i}/")