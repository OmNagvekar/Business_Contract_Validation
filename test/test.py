from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from ragas.metrics import  Faithfulness, FactualCorrectness,AspectCritic,ResponseRelevancy,SemanticSimilarity,AnswerCorrectness
from ragas import evaluate
from ragas import EvaluationDataset
import os
from ragas.run_config import RunConfig
import ast

# create account and get API KEY Here: https://app.ragas.io/
os.environ["RAGAS_APP_TOKEN"] = "API_KEY_Here" # Needed for uploading results to https://app.ragas.io/
my_run_config = RunConfig(max_workers=1, timeout=60)

def load_data(path):
    def safe_literal_eval(val):
      try:
          return ast.literal_eval(val)
      except (ValueError, SyntaxError):
          return []  # or some default value

    df = pd.read_csv(path)
    new_df = df[['Query', 'Responses', 'actual_response','retrieved_context']].copy()
    new_df.columns = ['user_input', 'response', 'reference','retrieved_contexts']
    # Convert boolean reference to string
    new_df['reference'] = new_df['reference'].astype(str)
    # Convert string representations of lists into actual lists
    new_df['retrieved_contexts'] = new_df['retrieved_contexts'].apply(safe_literal_eval)
    return new_df

def llm_initialization():
    config = {
        "model": "gemini-2.0-flash",  # or other model IDs
        "temperature": 0.4,
        "max_tokens": None,
        "top_p": 0.8,
    }
    with open("gemini_key.txt",'r') as f:
        key = f.read().strip()
    # Initialize with Google AI Studio
    evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        api_key=key
    ))
    embed_model = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name='BAAI/bge-base-en'))
    return evaluator_llm,embed_model

def evaluation_dataset():
    # Load the dataset
    df = load_data(r".\Final_Ragas_Evaluation_dataset_with_RAG_Responses.csv")
    # Convert to Ragas dataset schema
    dataset = EvaluationDataset.from_pandas(
        df
    )
    return dataset

def aspect_critic_score(evaluator_llm):
    metric = AspectCritic(
        name="Business_Contract_Clause_Standard_Evaluation",
        llm=evaluator_llm,
        definition=(
            "This metric measures the LLM's ability to determine whether a specific contract clause meets predefined legal and contractual standards. "
            "It evaluates if the clause is compliant by comparing it against key provisions and thresholds defined in standard templates, such as those "
            "derived from the Indian Contract Law and the Indian Contract Act. The evaluation should return a clear 'Yes' or 'No' along with a brief rationale "
            "that justifies the decision."
        )
    )
    return metric

def final_evaluation():
    
    # Load the dataset
    dataset = evaluation_dataset()
    
    # Initialize the LLM
    evaluator_llm,embed_model = llm_initialization()

    # Initialize the metric
    aspect_metrics = aspect_critic_score(evaluator_llm)

    # Evaluate the dataset using the metric
    results = evaluate(
        dataset=dataset, 
        metrics=[aspect_metrics, Faithfulness(), FactualCorrectness(),ResponseRelevancy(),SemanticSimilarity(),AnswerCorrectness()],
        llm=evaluator_llm,
        embeddings=embed_model,
        run_config=my_run_config,
        experiment_name='Business contract validation',
        show_progress=True
    )

    # Print the results
    print(results)
    # Save the results to a CSV file
    final_results = results.to_pandas()
    final_results.to_excel('final_results.xlsx', index=False)
    # Upload the results to Ragas
    try:
        results.upload()
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    final_evaluation()