# Business Contract Validation Project
![q](https://github.com/OmNagvekar/Business_Contract_Validation/assets/120325414/1ad46f38-11c7-49e6-b310-ba1de2025ef7)

## Team Black Pearl
1. Om Nagvekar  
2. Chinmay Hemant Bhosale  
3. Parth Sarnobat  
4. Mandar Patil  

---

## Presentation
 [here](https://docs.google.com/presentation/d/1oNmM9cjXkfXuyB23qdgomxTB6Q7wtNfU/edit?usp=drivesdk&ouid=106019895848480696926&rtpof=true&sd=true)

---

## Project Overview

The Business Contract Validation project is designed to automate the process of verifying whether business contracts adhere to predefined legal and contractual standards. The system classifies an input contract, compares its clauses against standard templates, and highlights any deviations. An evaluation module further measures the performance of the clause validation process using advanced metrics.

At its core, the project leverages modern machine learning, vector indexing, and natural language processing techniques. A TensorFlow-based classification model distinguishes between various contract types, while a vector-indexing mechanism using HuggingFace embeddings ensures that clause-level comparisons are both accurate and efficient. The system now integrates with the Gemini AOI (Gemini 2.0-Flash LLM) for clause evaluation, ensuring robust and real-time performance.

---

## Detailed Code Structure & Explanation

### 1. Classification and Embedding Generation

- **`document_classification.py`**  
  - **Purpose**: Extracts text from PDF contracts and generates embeddings using a pre-trained vectorizer (`vectorizer.pkl`).
  - **Process**:  
    1. Load the PDF into memory and extract text using PyMuPDF.
    2. Transform the text into feature embeddings with the vectorizer.
    3. Classify the document using a TensorFlow model (`Document_classification3.keras`), categorizing it into one of several predefined classes (e.g., Employment Agreements, Consulting Agreements, etc.).
  - **Key Points**:  
    - Robust error handling ensures that even if text extraction fails, the process returns a safe fallback.
    - Converts embeddings from sparse to dense format if necessary.

### 2. Vector Indexing and Document Comparison

- **`creating_indexes_and_storing.py`**  
  - **Purpose**: Creates vector indexes from standard documents that serve as benchmarks for clause comparison.
  - **Process**:  
    1. Scan the specified data directory containing standard contract documents.
    2. For each contract type (organized in subdirectories), create a vector index using HuggingFace embeddings.
    3. Store and reuse existing indexes to avoid unnecessary recomputation.
  - **Key Points**:  
    - Uses LlamaIndex’s `VectorStoreIndex` for efficient storage and retrieval.
    - Ensures rapid access to document representations during clause validation.

### 3. Clause Comparison & Query Processing

- **`query_processing.py`**  
  - **Purpose**: Splits an uploaded contract into clauses and evaluates each clause for compliance.
  - **Process**:  
    1. Split the PDF text into manageable chunks (clauses) using a custom text splitter.
    2. For each clause, generate a detailed prompt containing:
       - A project description.
       - An example evaluation that outlines the standard.
       - Context from previous clauses for continuity.
    3. Send the prompt to an LLM (using Gemini AOI of Gemini 2.0-Flash) for evaluation.
    4. Record the evaluation response and determine a Boolean compliance status.
    5. Highlight non-compliant clauses in the original PDF using PyMuPDF.
  - **Key Points**:  
    - Implements robust retry and rate-limiting for LLM API calls.
    - Aggregates responses and compliance scores for detailed reporting.

### 4. User Interface & PDF Highlighting

- **`ui.py`**  
  - **Purpose**: Provides an interactive Streamlit interface for users.
  - **Features**:  
    1. **PDF Upload**: Users can upload contract PDFs easily.
    2. **Processing Feedback**: Progress indicators and detailed logging keep users informed.
    3. **Results Display**: The processed PDF is shown with highlighted non-compliant clauses via an integrated PDF viewer.
    4. **Download Options**: Users can download both the annotated PDF and a CSV report detailing clause evaluations.
  - **Key Points**:  
    - Uses caching to minimize redundant processing.
    - Incorporates log cleanup to manage historical logs efficiently.

### 5. Evaluation Module

- **Evaluation Dataset Generation**:  
  - **Notebook**: `test/Evaluation_Dataset_Generation.ipynb`  
    Demonstrates how to create a synthetic evaluation dataset from contract clauses. The dataset includes fields such as `Clause`, `Truth_Value`, `Responses`, and the manually verified `actual_response`.

- **Evaluation Script**:  
  - **Script**: `test/test.py`  
    - **Purpose**: Evaluates the performance of the clause evaluation process using multiple metrics.
    - **Process**:
      1. **Data Loading**: Reads an evaluation CSV (converted to a standardized dataset).
      2. **LLM Initialization**: Sets up the Gemini AOI LLM via the ChatGoogleGenerativeAI interface (using API key from `gemini_key.txt`).
      3. **Metric Computation**:  
         Computes several metrics:
         - **Aspect Critic**: Measures the LLM’s ability to determine if a clause meets legal standards.
         - **Faithfulness**: Assesses if the response accurately reflects the clause without hallucinations.
         - **Factual Correctness**: Verifies that the evaluation aligns with established legal guidelines.
         - **Response Relevancy**: Evaluates how directly the response addresses the evaluation prompt.
         - **Semantic Similarity**: Compares the generated response with a reference answer.
         - **Answer Correctness**: A composite metric summarizing overall evaluation accuracy.
      4. **Results Storage**: Aggregates and saves the evaluation results as `final_results.xlsx` (and CSV) and optionally uploads them to an external service.
    - **Key Points**:
      - Utilizes the `ragas` and `langchain` libraries for metric computation.
      - Provides detailed insights into each clause’s evaluation performance.

---

## Evaluation Results

Below is a sample table of evaluation scores based on the contents of `final_results.csv`:

Okay, I can help you with that! Here is the table in Markdown format with the "Context Recall" column removed:

| Clause ID | Business_Contract_Clause_Standard_Evaluation | Faithfulness | Factual Correctness (F1) | Answer Relevancy | Semantic Similarity | Answer Correctness |
|-----------|----------------------------------------------:|-------------:|-------------------------:|-----------------:|--------------------:|-------------------:|
| 1         | 1.00                                          | 0.00         | 1.00                     | 0.81             | 0.88                | 0.22               |
| 2         | 0.00                                          | 0.50         | 0.00                     | 0.82             | 0.89                | 0.47               |
| 3         | 1.00                                          | 0.00         | 0.00                     | 0.86             | 0.83                | 0.21               |
| 4         | 0.00                                          | 0.00         | 0.53                     | 0.84             | 0.89                | 0.57               |
| 5         | 0.00                                          | 0.00         | 0.38                     | 0.83             | 0.86                | 0.21               |
| 6         | 0.00                                          | 0.00         | 0.80                     | 0.82             | 0.93                | 0.53               |
| 7         | 0.00                                          | 0.00         | 0.50                     | 0.85             | 0.91                | 0.64               |
| 8         | 1.00                                          | 0.00         | 0.18                     | 0.82             | 0.85                | 0.38               |
| 9         | 0.00                                          | 0.00         | 0.00                     | 0.82             | 0.87                | 0.30               |
| 10        | 1.00                                          | 0.00         | 0.00                     | 0.83             | 0.90                | 0.53               |

Let me know if you need any other modifications!

*Note: The above values are directly extracted from the sample rows of `final_results.csv`. In practice, the CSV may contain additional rows, and these metrics are computed for each evaluated clause.*

### Detailed Explanation of Evaluation Metrics

- **Business_Contract_Clause_Standard_Evaluation**:  
  Measures the LLM’s ability to accurately decide if a clause adheres to legal standards, returning 1 for adherence and 0 otherwise.

- **Context Recall**:  
  Indicates whether relevant context was successfully recalled during evaluation (0 if not recalled).

- **Faithfulness**:  
  Assesses if the response accurately reflects the clause content without introducing inaccuracies.

- **Factual Correctness (F1)**:  
  Measures the accuracy of the response against verifiable legal standards, reported as an F1 score.

- **Answer Relevancy**:  
  Evaluates how well the response directly addresses the evaluation prompt.

- **Semantic Similarity**:  
  Compares the semantic content of the generated response with a reference answer using similarity measures.

- **Answer Correctness**:  
  A composite score indicating the overall correctness of the LLM’s decision.

---

## Technologies Used

- **Machine Learning & TensorFlow**: For document classification.
- **Vector Indexing**: Using HuggingFace embeddings and LlamaIndex.
- **LLM Integration**: Utilizes Gemini AOI (Gemini 2.0-Flash) for clause evaluation (with local fallback via Ollama).
- **PDF Processing**: For text extraction and annotation using PyMuPDF.
- **Streamlit**: For the user interface.
- **Evaluation Framework**: Using libraries such as `ragas` and `langchain` to compute multiple evaluation metrics.

---

## Installation

To set up and run the project locally:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/OmNagvekar/Business_Contract_Validation.git
    cd business-contract-validation
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Classification Model and Vectorizer**:
    - Place the TensorFlow [model](https://huggingface.co/Chinya/Document_Classification/) (`Document_classification3.keras`) and the vectorizer (`vectorizer.pkl`) in the repository root.

4. **Configure Gemini API**:
    - Ensure you have your Gemini AOI API key saved in `gemini_key.txt`.

5. **Update Paths (if necessary)**:
    - Adjust file paths in the code to match your local environment.

6. **Run the Streamlit Application**:
    ```bash
    streamlit run ui.py
    ```

7. **Run the Evaluation Module**:
    - From the `test` directory, run:
      ```bash
      python test/test.py
      ```
    - The evaluation results will be saved as `final_results.xlsx` and printed to the console.

---

## Usage

1. **Contract Processing**:  
   - Upload a PDF contract through the Streamlit interface.
   - The system classifies the document, evaluates each clause using Gemini AOI, and highlights any deviations.
   - Download options are provided for the highlighted PDF and a CSV report with detailed evaluations.

2. **Evaluation**:  
   - Run the evaluation script to generate an evaluation dataset, compute multiple performance metrics, and review detailed scores saved in `final_results.xlsx`.
   - The evaluation metrics help identify areas for improvement and provide a comprehensive performance overview.

---

## Data Sources

Some contracts used in this project were sourced from the [Atticus Open Contract Dataset (AOK Beta)](https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta/code) on Kaggle.

---

## Contributing

We welcome contributions to improve the project. Please fork the repository, implement your changes, and submit a pull request. Contributions in areas such as model training, evaluation metrics, or UI enhancements are especially appreciated.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any queries or support, please contact [omnagvekar29@gmail.com, chinmayhbhosale02@gmail.com].

---