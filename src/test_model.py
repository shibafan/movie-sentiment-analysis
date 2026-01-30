import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/distilbert_imdb_binary"
MAX_LENGTH = 2000


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_text(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    predicted_class = int(probabilities.argmax())
    sentiment = "positive" if predicted_class == 1 else "negative"
    positive_prob = float(probabilities[1])
    negative_prob = float(probabilities[0])
    
    return sentiment, positive_prob, negative_prob


def main():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("Enter a review:  ")
    print()

    while True:
        text = input("> ").strip()
        if not text:
            break

        sentiment, positive_prob, negative_prob = predict_text(text, tokenizer, model, device)
        print(f"Sentiment: {sentiment}")
        print(f"Positive: {positive_prob:.3f}, Negative: {negative_prob:.3f}")
        print()


if __name__ == "__main__":
    main()
