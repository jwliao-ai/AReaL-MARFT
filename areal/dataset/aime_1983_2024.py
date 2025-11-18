import pandas as pd
from datasets import Dataset


def get_aime_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """
    Load AIME dataset from CSV file for supervised fine-tuning.
    
    CSV columns: ID, Year, Problem Number, Question, Answer, Part
    """
    # Load CSV file
    df = pd.read_csv(path)
    
    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    def process(sample):
        # Construct input sequence: question + answer
        question = sample["Question"]
        answer = str(sample["Answer"])
        
        # Format the answer with \boxed{} for consistency with other math datasets
        formatted_answer = f"\n\nThe answer is \\boxed{{{answer}}}{tokenizer.eos_token}"
        
        seq_token = tokenizer.encode(question + formatted_answer)
        prompt_token = tokenizer.encode(question)
        
        # Loss mask: 0 for prompt, 1 for answer
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        
        return {"input_ids": seq_token, "loss_mask": loss_mask}
    
    # Process dataset and remove original columns
    dataset = dataset.map(process).remove_columns(
        ["ID", "Year", "Problem Number", "Question", "Answer", "Part"]
    )
    
    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
    
    return dataset


def get_aime_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """
    Load AIME dataset from CSV file for reinforcement learning.
    
    CSV columns: ID, Year, Problem Number, Question, Answer, Part
    """
    # Load CSV file
    df = pd.read_csv(path)
    
    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    def process(sample):
        question = sample["Question"]
        
        messages = [
            {
                "role": "user",
                "content": question,
            }
        ]
        
        # Store the ground truth answer for evaluation
        return {
            "messages": messages,
            "answer": str(sample["Answer"]),
            "year": sample["Year"],
            "problem_number": sample["Problem Number"],
        }
    
    # Process dataset and remove the Question column (keep metadata for evaluation)
    dataset = dataset.map(process).remove_columns(["ID", "Question", "Answer", "Part"])
    
    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:
        
        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length
        
        dataset = dataset.filter(filter_length)
    
    return dataset
