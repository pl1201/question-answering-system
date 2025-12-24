import argparse
import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.data import squad_json_to_dataframe, make_splits, to_hf_datasets
from src.postprocess import prepare_features_fn, postprocess_predictions_fixed_final


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model on SQuAD validation split")
    parser.add_argument("--train_file", type=str, required=True, help="Path to SQuAD train JSON (v1.1)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--train_size", type=int, default=2000, help="Train subset size (same as used in training)")
    parser.add_argument("--valid_size", type=int, default=500, help="Validation subset size")
    return parser.parse_args()


def main():
    args = parse_args()

    df = squad_json_to_dataframe(args.train_file)
    _, valid_df = make_splits(df, train_size=args.train_size, valid_size=args.valid_size)
    # Only need validation set for eval
    from datasets import Dataset

    valid_ds = Dataset.from_pandas(valid_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prepare_features = prepare_features_fn(tokenizer, max_length=512, doc_stride=128)
    valid_tokenized = valid_ds.map(prepare_features, batched=True, remove_columns=valid_ds.column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)

    # Predict
    from transformers import Trainer

    trainer = Trainer(model=model, tokenizer=tokenizer)
    raw_preds = trainer.predict(valid_tokenized)

    preds, refs = postprocess_predictions_fixed_final(
        valid_df.to_dict("records"), valid_tokenized, raw_preds.predictions, tokenizer
    )

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=preds, references=refs)
    print(results)


if __name__ == "__main__":
    main()

