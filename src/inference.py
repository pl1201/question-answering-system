from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


def load_qa_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def predict_answer(model, tokenizer, question: str, context: str, device: str = "cpu", max_length: int = 512):
    model.eval()
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

    # Limit to context tokens
    sep_idx = (inputs["input_ids"][0] == tokenizer.sep_token_id).nonzero()[0].item()
    context_start = sep_idx + 1
    context_end = inputs["input_ids"].shape[1] - 1

    start_rel = torch.argmax(start_logits[context_start:context_end]).item()
    end_rel = torch.argmax(end_logits[context_start:context_end]).item()
    start_idx = context_start + start_rel
    end_idx = context_start + end_rel
    if end_idx < start_idx:
        end_idx = start_idx

    pred = tokenizer.decode(inputs["input_ids"][0][start_idx : end_idx + 1], skip_special_tokens=True).strip()
    return pred



