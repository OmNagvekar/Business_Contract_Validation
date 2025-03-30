import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
import keras

import pymupdf as pdf
import logging

logger  = logging.getLogger(__name__)

class DocumentClassification:
    def __init__(self,path,model) -> None:
        logger.info("Initializing DocumentClassification for path: %s", path)
        self.path = path
        self.feature_array = []
        try:
            with open("./vectorizer.pkl", 'rb') as file:
                self.vectorizer = pickle.load(file)
            logger.info("Vectorizer loaded successfully.")
        except Exception as e:
            logger.error("Failed to load vectorizer: %s", e)
            raise e
        self.model = model#tf.keras.models.load_model("/home/chinu_tensor/Business_Contract_Validation/Document_classification3.keras")
        logger.info("Model loaded successfully.")
        self.text = self.pdf_loader()
        logger.info("PDF text loaded. Length of text: %d characters.", len(self.text[0]) if self.text and self.text[0] else 0)
        self.embed_gen()    
        logger.info("Embedding generation completed.")

    def embed_gen(self):
        try:
            logger.info("Generating embeddings using vectorizer.")
            feature = self.vectorizer.transform(self.text)
            self.feature_array = feature
            if isinstance(self.feature_array, tf.SparseTensor):
                logger.info("Converting sparse tensor to dense.")
                self.feature_array = tf.sparse.to_dense(self.feature_array)
            logger.info("Embeddings generated successfully.")
        except Exception as e:
            logger.error("Error during embedding generation: %s", e)
            raise e
    
    def classify_doc(self):
        try:
            logger.info("Running model prediction for document classification.")
            prediction = self.model.predict(self.feature_array)
            prediction = np.argmax(prediction,axis=1)
            labels = ['Affiliate_Agreements','Agency Agreements','Collaboration','Consulting Agreements','Co_Branding','Development','Distributor','Employment Agreements','Endorsement','Franchise',
                        'Hosting','Intellectual Agreement','IP','Joint Venture','License_Agreements','Loan Agreement','Maintenance','Manufacturing','Marketing','Non_Compete_Non_Solicit','Outsourcing',
                        'Promotion','Reseller','Service','Severance Agreement','Sponsorship','Strategic Alliance','Supply','Transportation','Warrent Agreement'
                    ]
            label=labels[prediction[0]]
            logger.info("Document classified as: %s", label)
            return label
        except Exception as e:
            logger.error("Error during document classification: %s", e)
            raise e

    def pdf_loader(self):
        try:
            logger.info("Loading PDF from path: %s", self.path)
            all_text = ""
            doc = pdf.open(self.path)
            logger.info("PDF opened successfully. Number of pages: %d", len(doc))
            for page in doc:
                all_text += page.get_text()
            logger.info("Completed extracting text from PDF.")
            return [all_text]
        except Exception as e:
            logger.error("Error loading PDF: %s", e)
            return [""]
if __name__=="__main__":
    doc = DocumentClassification('../Downloads/exhibit101.pdf')
    print(doc.classify_doc())
