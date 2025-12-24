import numpy as np
from typing import Tuple, List, Dict, Any


def prepare_features_fn(tokenizer, max_length=512, doc_stride=128):
    """
    Return a preprocessing function for datasets.map that:
    - tokenizes question/context with overflow for long contexts
    - computes start/end positions using offset mapping
    """

    def _prepare_features(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = examples["answer"][sample_idx]
            context = examples["context"][sample_idx]

            start_char = context.find(answer)
            if start_char == -1:
                start_char = context.lower().find(answer.lower())

            end_char = start_char + len(answer) if start_char != -1 else -1

            sequence_ids = tokenized.sequence_ids(i)

            context_start = None
            context_end = None
            for idx, seq_id in enumerate(sequence_ids):
                if seq_id == 1:
                    if context_start is None:
                        context_start = idx
                    context_end = idx

            start_pos = end_pos = 0

            if start_char != -1 and context_start is not None:
                context_char_start = offsets[context_start][0] if offsets[context_start][0] is not None else 0
                context_char_end = offsets[context_end][1] if offsets[context_end][1] is not None else len(context)

                if context_char_start <= start_char < context_char_end and context_char_start < end_char <= context_char_end:
                    for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
                        if seq_id != 1:
                            continue
                        if offset[0] is None or offset[1] is None:
                            continue
                        if start_pos == 0 and offset[0] <= start_char < offset[1]:
                            start_pos = idx
                        if offset[0] < end_char <= offset[1]:
                            end_pos = idx
                            break
                    if end_pos < start_pos:
                        end_pos = start_pos

            start_positions.append(start_pos)
            end_positions.append(end_pos)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    return _prepare_features


def postprocess_predictions_fixed_final(
    examples: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    tokenizer,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """
    Post-process logits to text answers. Returns (predictions, references)
    with references containing answer_start required by SQuAD metric.
    """
    start_logits, end_logits = raw_predictions

    example_to_features = {}
    for i, feature in enumerate(features):
        example_idx = feature.get("overflow_to_sample_mapping", i)
        example_to_features.setdefault(example_idx, []).append(i)

    predictions = []
    references = []

    for example_idx, example in enumerate(examples):
        example_id = example.get("id", example_idx)
        answer_text = example.get("answer", "")
        context_text = example.get("context", "")

        feature_indices = example_to_features.get(example_idx, [example_idx])
        valid_predictions = []

        for feature_idx in feature_indices:
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]

            start_indices = np.argsort(start_logit)[-n_best_size:][::-1]
            end_indices = np.argsort(end_logit)[-n_best_size:][::-1]

            feature = features[feature_idx]
            input_ids = feature["input_ids"]
            sequence_ids = feature.get("sequence_ids", None)

            if sequence_ids is None:
                sep_idx = None
                for idx, token_id in enumerate(input_ids):
                    if token_id == tokenizer.sep_token_id:
                        sep_idx = idx
                        break
                if sep_idx:
                    sequence_ids = [0] * (sep_idx + 1) + [1] * (len(input_ids) - sep_idx - 1)
                else:
                    sequence_ids = [1] * len(input_ids)

            context_start_idx = None
            context_end_idx = None
            for idx, seq_id in enumerate(sequence_ids):
                if seq_id == 1:
                    if context_start_idx is None:
                        context_start_idx = idx
                    context_end_idx = idx

            for start_idx in start_indices:
                for end_idx in end_indices:
                    if start_idx > end_idx:
                        continue
                    if context_start_idx and (start_idx < context_start_idx or end_idx > context_end_idx):
                        continue
                    if end_idx - start_idx + 1 > max_answer_length:
                        continue

                    score = start_logit[start_idx] + end_logit[end_idx]
                    answer_tokens = input_ids[start_idx : end_idx + 1]
                    decoded_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                    if len(decoded_answer) < 1:
                        continue
                    valid_predictions.append(
                        {"text": decoded_answer, "score": float(score), "start": int(start_idx), "end": int(end_idx)}
                    )

        if valid_predictions:
            best_pred = max(valid_predictions, key=lambda x: x["score"])
            pred_text = best_pred["text"]
        else:
            pred_text = ""

        predictions.append({"id": example_id, "prediction_text": pred_text})

        answer_start = context_text.find(answer_text)
        if answer_start == -1:
            answer_start = context_text.lower().find(answer_text.lower())

        references.append(
            {
                "id": example_id,
                "answers": {
                    "text": [answer_text],
                    "answer_start": [answer_start] if answer_start != -1 else [],
                },
            }
        )

    return predictions, references



