from datasets import load_dataset

def get_math_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    if path.endswith(".json") or path.endswith(".jsonl"):
        # 修复：显式指定 split 名称对应的文件，防止 load_dataset 默认归类为 'train' 导致 split='test' 时报错
        data_files = {split: path} if split else path
        dataset = load_dataset("json", data_files=data_files, split=split)
    else:
        dataset = load_dataset(path, split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["problem"] + sample["solution"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["problem"])
        
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(
        ["problem", "solution", "answer", "subject", "level", "unique_id"]
    )

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_math_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    if path.endswith(".json") or path.endswith(".jsonl"):
        # 修复：显式指定 split 名称对应的文件，防止 load_dataset 默认归类为 'train' 导致 split='test' 时报错
        data_files = {split: path} if split else path
        dataset = load_dataset("json", data_files=data_files, split=split)
    else:
        dataset = load_dataset(path, split=split)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["problem"]
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["problem"])

    if max_length is not None:
        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset