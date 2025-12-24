import json
import pandas as pd
from datasets import Dataset


def squad_json_to_dataframe(file_path: str) -> pd.DataFrame:

    file = json.loads(open(file_path, "r", encoding="utf-8").read())
    records = []
    for article in file["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                for ans in qa["answers"]:
                    records.append(
                        {
                            "id": qa["id"],
                            "question": question,
                            "context": context,
                            "answer": ans["text"],
                        }
                    )
    return pd.DataFrame(records)


def make_splits(df: pd.DataFrame, train_size: int = 2000, valid_size: int = 500):
    """
    Shuffle and split the DataFrame into train/valid subsets.
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df.iloc[:train_size]
    valid_df = df.iloc[train_size : train_size + valid_size]
    return train_df, valid_df


def to_hf_datasets(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """
    Convert pandas DataFrames to HuggingFace Dataset objects.
    """
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)
    return train_ds, valid_ds



