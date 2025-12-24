import argparse
import os
import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments

from src.data import squad_json_to_dataframe, make_splits, to_hf_datasets
from src.postprocess import prepare_features_fn, postprocess_predictions_fixed_final
from src.callbacks import EarlyStoppingCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train QA model on SQuAD")
    parser.add_argument("--train_file", type=str, required=True, help="Path to SQuAD train JSON (v1.1)")
    parser.add_argument("--output_dir", type=str, default="./qa_model", help="Output directory for checkpoints")
    parser.add_argument("--train_size", type=int, default=2000, help="Number of training examples to use")
    parser.add_argument("--valid_size", type=int, default=500, help="Number of validation examples to use")
    parser.add_argument("--model_name", type=str, default="albert-base-v2", help="HF model name or path")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load and split data
    df = squad_json_to_dataframe(args.train_file)
    train_df, valid_df = make_splits(df, train_size=args.train_size, valid_size=args.valid_size)
    train_ds, valid_ds = to_hf_datasets(train_df, valid_df)

    # 2) Tokenizer & preprocess
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prepare_features = prepare_features_fn(tokenizer, max_length=512, doc_stride=128)
    train_tokenized = train_ds.map(prepare_features, batched=True, remove_columns=train_ds.column_names)
    valid_tokenized = valid_ds.map(prepare_features, batched=True, remove_columns=valid_ds.column_names)

    # 3) Model
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # 4) Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        logging_dir=os.path.join(args.output_dir, "logs"),
        seed=42,
        dataloader_num_workers=4,
        report_to="none",
    )

    # 5) Metrics
    squad_metric = evaluate.load("squad")

    def compute_metrics(eval_pred):
        predictions, _labels = eval_pred
        preds, refs = postprocess_predictions_fixed_final(
            valid_df.to_dict("records"), valid_tokenized, predictions, tokenizer
        )
        return squad_metric.compute(predictions=preds, references=refs)

    # 6) Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(patience=2, min_delta=0.001)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Final eval
    eval_results = trainer.evaluate()
    print("Final evaluation:", eval_results)


if __name__ == "__main__":
    main()



