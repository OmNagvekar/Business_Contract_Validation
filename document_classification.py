import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
import keras

import pymupdf as pdf

class DocumentClassification:
    def __init__(self,path) -> None:
        self.path = path
        self.feature_array = []
        file=open("./vectorizer.pkl",'rb')
        self.vectorizer = pickle.load(file)
        file.close()
        self.model = tf.keras.models.load_model("./Document_classification3.keras")
        self.text = self.pdf_loader()
        self.embed_gen()    

    def embed_gen(self):
        feature = self.vectorizer.transform(self.text)
        self.feature_array = feature
        if isinstance(self.feature_array, tf.SparseTensor):
          self.feature_array = tf.sparse.to_dense(self.feature_array)
    
    def classify_doc(self):
        prediction = self.model.predict(self.feature_array)
        prediction = np.argmax(prediction,axis=1)
        labels = ['Affiliate_Agreements','Agency Agreements','Collaboration','Consulting Agreements','Co_Branding','Development','Distributor','Employment Agreements','Endorsement','Franchise',
                    'Hosting','Intellectual Agreement','IP','Joint Venture','License_Agreements','Loan Agreement','Maintenance','Manufacturing','Marketing','Non_Compete_Non_Solicit','Outsourcing',
                    'Promotion','Reseller','Service','Severance Agreement','Sponsorship','Strategic Alliance','Supply','Transportation','Warrent Agreement'
                ]
        return labels[prediction[0]]

    def pdf_loader(self):
        all_text = ""
        doc = pdf.open(self.path)
        for page in doc:
            all_text += page.get_text()
        return [all_text]