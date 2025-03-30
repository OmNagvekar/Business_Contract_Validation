import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class CreateVectorIndex:
    def __init__(self, base_path: str) -> None:
        """
        Initializes the vector index creator.

        :param base_path: The base path containing the 'data' directory.
        """
        logger.info("Initializing CreateVectorIndex with base_path: %s", base_path)
        self.base_path = base_path
        self.data_path = os.path.join(base_path, "data")
        self.index_dir = os.path.join(base_path, "vector_indexes")
        os.makedirs(self.index_dir, exist_ok=True)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")
    
    def create_indexes(self) -> None:
        """
        Creates vector indexes for each subdirectory within the data directory.
        Skips subdirectories that already have a non-empty index folder.
        """
        logger.info("Starting to create indexes from data path: %s", self.data_path)
        if not os.path.exists(self.data_path):
            logger.error("Data directory not found: %s", self.data_path)
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        items = os.listdir(self.data_path)
        for item in tqdm(items, desc="Creating vector indexes"):
            item_path = os.path.join(self.data_path, item)
            # Process only directories
            if os.path.isdir(item_path):
                index_item_path = os.path.join(self.index_dir, item)
                # Check if index already exists and is non-empty
                if os.path.exists(index_item_path) and os.listdir(index_item_path):
                    logger.info("Index for '%s' already exists. Skipping.", item)
                    continue
                
                os.makedirs(index_item_path, exist_ok=True)
                logger.info("Creating index for directory: %s", item)
                docs = SimpleDirectoryReader(input_dir=item_path).load_data()
                index = VectorStoreIndex(docs, embed_model=self.embed_model)
                index.storage_context.persist(persist_dir=index_item_path)
                logger.info("Index created and persisted for directory: %s", item)
            else:
                logger.warning("Skipping non-directory item: %s", item)

        logger.info("Vector indexes created successfully.")