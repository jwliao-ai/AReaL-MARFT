from datasets import load_dataset
import json

def get_codeforces_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """
    用于处理 Codeforces SFT 数据集。
    假设数据集中包含 'solutions' (代码列表) 或 'solution' (代码字符串) 字段作为 target。
    """
    if path.endswith(".json") or path.endswith(".jsonl"):
        # 修复：显式指定 split 名称对应的文件，防止 load_dataset 默认归类为 'train' 导致 split='test' 时报错
        data_files = {split: path} if split else path
        dataset = load_dataset("json", data_files=data_files, split=split)
    else:
        dataset = load_dataset(path, split=split)

    def process(sample):
        if "prompt" in sample and isinstance(sample["prompt"], list) and len(sample["prompt"]) > 0:
            question = sample["prompt"][0]["content"]
        elif "extra_info" in sample and "question" in sample["extra_info"]:
            question = sample["extra_info"]["question"]
        else:
            question = ""

        solution = ""
        if "solutions" in sample and sample["solutions"]:
            solution = sample["solutions"][0] if isinstance(sample["solutions"], list) else sample["solutions"]
        elif "solution" in sample:
            solution = sample["solution"]
        
        seq_token = tokenizer.encode(
            question + "\n```python\n" + solution + "\n```" + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(question + "\n```python\n")
        
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    cols_to_remove = [c for c in dataset.column_names if c not in ["input_ids", "loss_mask"]]
    dataset = dataset.map(process).remove_columns(cols_to_remove)

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_codeforces_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """
    用于处理 Codeforces RL 数据集。
    保留问题 Prompt，同时保留用于 Reward Model 评测的 metadata (如 test cases)。
    """
    if path.endswith(".json") or path.endswith(".jsonl"):
        # 修复：显式指定 split 名称对应的文件，防止 load_dataset 默认归类为 'train' 导致 split='test' 时报错
        data_files = {split: path} if split else path
        dataset = load_dataset("json", data_files=data_files, split=split)
    else:
        dataset = load_dataset(path, split=split)

    TEMPLATE_HEADER = "You will be given a problem statement, test case constraints and example test inputs and outputs. Please reason step by step about the solution (that must respect memory and time limits), then provide a complete implementation in python3.\n\nYour solution must read input from standard input (cin), write output to standard output (cout).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```python\n<your code here>\n```\n"
    TEMPLATE_FOOTER = "\nNow solve the problem and return the code.\n"

    def process(sample):
        content = ""
        if "prompt" in sample and isinstance(sample["prompt"], list) and len(sample["prompt"]) > 0:
            content = sample["prompt"][0]["content"]
        elif "extra_info" in sample and "question" in sample["extra_info"]:
            content = sample["extra_info"]["question"]

        if content:
            content = content.replace(TEMPLATE_HEADER, "").replace(TEMPLATE_FOOTER, "").strip()

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        answer = {}
        if "reward_model" in sample and "ground_truth" in sample["reward_model"]:
            answer = sample["reward_model"]["ground_truth"]

        return {"messages": messages, "answer": answer}

    dataset = dataset.map(process)

    cols_to_keep = {"messages", "answer"}
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    dataset = dataset.remove_columns(cols_to_remove)

    if max_length is not None:
        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset