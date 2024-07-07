# Business Contract Validation Project
## Team Black Pearl
    1.Anuja Deshpande.
    2.Arya Dhavale.
    3.Chinmay Hemant Bhosale.
    4.Om Nagvekar.
    5.Sanket Jadhav.
## Problem Statement
The aim of this project is to automate the validation of business contracts by ensuring they adhere to predefined standards. This is achieved by classifying the input contracts and then comparing them against standard documents to identify any deviations.

## Approach

### 1. Classification of Input Contracts
- **Model Training**: We trained a classification model on a dataset of contracts to categorize input contracts into predefined classes.
- **Input Classification**: When a new contract is submitted, the model classifies it into one of the specified classes.
- **Model Access**: The classification model is too large to include directly in this repository. It is uploaded on HuggingFace and can be accessed [here](https://huggingface.co/Chinya/Document_Classification/). Note that the model is essential for the classification step to function correctly.

### 2. Vector Indexing and Comparison
- **Standard Document Indexing**: We created vector indexes for each standard document, which will be used for comparison.
- **Loading Appropriate Index**: Based on the classification of the input contract, the corresponding vector index of the standard document is loaded.

### 3. Clause Comparison
- **Splitting into Clauses**: The input contract is split into individual clauses for detailed analysis.
- **Clause Validation**: Each clause is compared with the corresponding clauses in the standard document using the LlamaIndex and Llama 2 Chat HuggingFace model.

### 4. User Interface
- **Streamlit**: We utilized Streamlit to create an interactive and user-friendly interface.
  - **PDF Processing**: The UI allows users to upload a PDF contract, which is then processed and analyzed.
  - **Highlighting Deviations**: The processed PDF is displayed with highlighted clauses where deviations from the standard are detected.
  - **Download Option**: Users can download the highlighted PDF along with the responses generated for specific clauses.

### 5. Fast Processing with Gradient API
- **Gradient API**: For fast processing, we have used the Gradient API in LlamaIndex. If appropriate computational resources are available, users can run the local LLM using Ollama or HuggingFace. The code for running locally is commented in the project files.
- **API Key**: To use the Gradient API, obtain an API key [here](https://auth.gradient.ai).

## Features
- **Automated Classification**: Efficiently classifies contracts into predefined categories.
- **Accurate Validation**: Uses advanced models to ensure precise comparison with standard documents.
- **Interactive UI**: Provides an easy-to-use interface for uploading and reviewing contracts.
- **Highlight Deviations**: Clearly marks clauses that deviate from the standard, aiding in quick identification.
- **Downloadable Results**: Offers the option to download the analyzed and highlighted contract PDF.

## Technologies Used
- **Machine Learning**: For contract classification.
- **Vector Indexing**: To store and compare standard documents.
- **LlamaIndex & Llama 2 Chat HF Model**: For clause validation.
- **Streamlit**: For the user interface.
- **PDF Processing**: For handling and highlighting contract clauses.
- **Gradient API**: For fast processing.

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/OmNagvekar/Business_Contract_Validation.git
    cd business-contract-validation
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Classification Model**:
    Download the classification model from HuggingFace [here](https://huggingface.co/Chinya/Document_Classification/) and place it in the appropriate directory as specified in the code.

4. **Update Paths**:
    Update the paths in the code according to your machine setup.

5. **Run the Streamlit Application**:
    ```bash
    streamlit run ui.py
    ```

## Usage
1. **Upload Contract**: Use the Streamlit interface to upload the contract PDF.
2. **Classification and Validation**: The system will classify the contract and validate each clause against the standard.
3. **Review Results**: View the processed contract with highlighted deviations in the Streamlit interface.
4. **Download**: Download the highlighted PDF along with the responses for specific clauses.

## Data
Some of the contracts used in this project were sourced from the [Atticus Open Contract Dataset (AOK Beta)](https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta/code) on Kaggle.

## Contributing
We welcome contributions to improve the project. Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

For any queries or support, please contact [omnagvekar29@gmail.com].

---

This README provides an overview of the Business Contract Validation project, detailing the problem statement, approach, features, technologies used, installation steps, and usage instructions.
