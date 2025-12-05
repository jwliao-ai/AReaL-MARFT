import json
import zlib
import pickle
import base64
from datasets import load_dataset


def _decode_test_cases(test_cases_str: str) -> list:
    """
    解码 test cases，支持 JSON 格式和压缩格式。
    """
    try:
        return json.loads(test_cases_str)
    except:
        return json.loads(
            pickle.loads(
                zlib.decompress(
                    base64.b64decode(test_cases_str.encode("utf-8"))
                )
            )
        )


def _build_question_content(sample: dict) -> str:
    """
    从 LiveCodeBench 样本构建问题内容。
    """
    question_title = sample.get("question_title", "")
    question_content = sample.get("question_content", "")
    starter_code = sample.get("starter_code", "")
    
    # 构建完整的问题描述
    content = f"# {question_title}\n\n{question_content}"
    
    # 如果有 starter code，添加到问题中
    if starter_code:
        content += f"\n\n## Starter Code\n```python\n{starter_code}\n```"
    
    return content


def get_livecodebench_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """
    用于处理 LiveCodeBench SFT 数据集。
    LiveCodeBench 数据集是 jsonl 格式，包含 question_title, question_content, starter_code 等字段。
    """
    if path.endswith(".json") or path.endswith(".jsonl"):
        data_files = {split: path} if split else path
        dataset = load_dataset("json", data_files=data_files, split=split)
    else:
        dataset = load_dataset(path, split=split)

    def process(sample):
        # 构建问题内容
        question = _build_question_content(sample)
        
        # 获取解决方案（如果有的话）
        solution = ""
        if "solution" in sample and sample["solution"]:
            solution = sample["solution"]
        elif "solutions" in sample and sample["solutions"]:
            solution = sample["solutions"][0] if isinstance(sample["solutions"], list) else sample["solutions"]
        
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


def get_livecodebench_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """
    用于处理 LiveCodeBench RL 数据集。
    保留问题 Prompt，同时保留用于 Reward Model 评测的 metadata (如 test cases)。
    
    LiveCodeBench 数据格式:
    - question_title: 问题标题
    - question_content: 问题内容
    - platform: 平台 (atcoder, codeforces, leetcode)
    - question_id: 问题 ID
    - contest_id: 比赛 ID
    - contest_date: 比赛日期
    - starter_code: 起始代码
    - difficulty: 难度 (easy, medium, hard)
    - public_test_cases: 公开测试用例 (JSON 格式)
    - private_test_cases: 私有测试用例 (可能是压缩格式)
    - metadata: 元数据
    """
    if path.endswith(".json") or path.endswith(".jsonl"):
        data_files = {split: path} if split else path
        dataset = load_dataset("json", data_files=data_files, split=split)
    else:
        dataset = load_dataset(path, split=split)

    TEMPLATE_HEADER = "You will be given a problem statement, test case constraints and example test inputs and outputs. Please reason step by step about the solution (that must respect memory and time limits), then provide a complete implementation in python3.\n\nYour solution must read input from standard input (cin), write output to standard output (cout).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```python\n<your code here>\n```\n"
    TEMPLATE_FOOTER = "\nNow solve the problem and return the code.\n"

    def process(sample):
        # 构建问题内容
        question_title = sample.get("question_title", "")
        question_content = sample.get("question_content", "")
        starter_code = sample.get("starter_code", "")
        platform = sample.get("platform", "")
        difficulty = sample.get("difficulty", "")
        
        # 解析公开测试用例用于展示
        public_test_cases = []
        if "public_test_cases" in sample and sample["public_test_cases"]:
            try:
                public_test_cases = json.loads(sample["public_test_cases"]) if isinstance(sample["public_test_cases"], str) else sample["public_test_cases"]
            except:
                pass
        
        # 构建完整的问题描述
        content = f"# {question_title}\n\n"
        content += f"**Platform:** {platform}\n"
        content += f"**Difficulty:** {difficulty}\n\n"
        content += f"## Problem\n{question_content}\n"
        
        # 添加示例测试用例
        if public_test_cases:
            content += "\n## Example Test Cases\n"
            for i, tc in enumerate(public_test_cases):
                content += f"\n### Test Case {i + 1}\n"
                content += f"**Input:**\n```\n{tc.get('input', '')}\n```\n"
                content += f"**Output:**\n```\n{tc.get('output', '')}\n```\n"
        
        # 如果有 starter code
        if starter_code:
            content += f"\n## Starter Code\n```python\n{starter_code}\n```\n"
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # 构建 answer 字典，只包含测试用例 (inputs 和 outputs)
        answer = {"inputs": [], "outputs": []}
        if "public_test_cases" in sample and sample["public_test_cases"]:
            try:
                parsed_tests = json.loads(sample["public_test_cases"]) if isinstance(sample["public_test_cases"], str) else sample["public_test_cases"]
                for tc in parsed_tests:
                    answer["inputs"].append(tc.get("input", ""))
                    answer["outputs"].append(tc.get("output", ""))
            except:
                pass
        
        # 构建 metadata 字典，包含其他评测所需的信息
        metadata = {
            "question_id": sample.get("question_id", ""),
            "contest_id": sample.get("contest_id", ""),
            "platform": platform,
            "difficulty": difficulty,
            "starter_code": starter_code,
        }
        
        # 解析并保存私有测试用例到 metadata
        if "private_test_cases" in sample and sample["private_test_cases"]:
            try:
                metadata["private_test_cases"] = _decode_test_cases(sample["private_test_cases"])
            except:
                metadata["private_test_cases"] = []
        
        # 保存原始 metadata
        if "metadata" in sample and sample["metadata"]:
            try:
                metadata["original_metadata"] = json.loads(sample["metadata"]) if isinstance(sample["metadata"], str) else sample["metadata"]
            except:
                metadata["original_metadata"] = {}

        return {"messages": messages, "answer": answer, "metadata": metadata}

    dataset = dataset.map(process)

    cols_to_keep = {"messages", "answer", "metadata"}
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    dataset = dataset.remove_columns(cols_to_remove)

    if max_length is not None:
        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
