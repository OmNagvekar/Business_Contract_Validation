import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core  import VectorStoreIndex
from tqdm import tqdm
class CreateVectorIndex:
    def __init__(self,path:str) -> None:
        self.path=path
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")
        self.exists=self.create_index(self.path)
    
    def create_index(self,path:str):
        if not os.path.exists('./vector_indexes'):
            files = os.listdir("./data")
            for i in tqdm(files):
                doc = SimpleDirectoryReader(input_dir=f"{path}/data/{i}").load_data()
                os.mkdir(f"{path}/vector_indexes/{i}")
                index = VectorStoreIndex(doc,embed_model=self.embed_model)
                index.storage_context.persist(persist_dir = f"{path}/vector_indexes/{i}/")
            return True
        else:
            return False