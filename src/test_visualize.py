import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


MODEL_DIR = "models/distilbert_imdb_binary"
DATA_DIR = "data/imdb_binary"
MAX_LENGTH = 256
BATCH_SIZE = 32


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(path):
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns: text, label. Found: {list(df.columns)}")
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def predict(df, tokenizer, model, device):
    ds = Dataset.from_pandas(df[["text", "label"]])

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds_tok = ds.map(tok, batched=True, remove_columns=["text"])
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    def collate_fn(batch):
        # Extract labels before collation
        labels = torch.stack([item["label"] for item in batch])
        # Remove labels for the padding collator
        batch_without_labels = [{k: v for k, v in item.items() if k != "label"} for item in batch]
        # Apply padding collator
        collated = DataCollatorWithPadding(tokenizer=tokenizer)(batch_without_labels)
        # Add labels back
        collated["label"] = labels
        return collated

    dl = DataLoader(ds_tok, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_probs = []
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dl:
            labels = batch["label"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items() if k != "label"}

            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

    probs = np.vstack(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    return labels, preds, probs


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm)  # default colormap
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["NEG", "POS"])
    plt.yticks([0, 1], ["NEG", "POS"])

    # write counts on the matrix
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_confidence_hist(confidence, correct_mask, title):
    plt.figure(figsize=(6, 4))

    plt.hist(confidence[correct_mask], bins=30, alpha=0.7, label="Correct")
    plt.hist(confidence[~correct_mask], bins=30, alpha=0.7, label="Incorrect")

    plt.title(title)
    plt.xlabel("Model confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_split(name, df, tokenizer, model, device):
    labels, preds, probs = predict(df, tokenizer, model, device)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    confidence = probs.max(axis=1)
    correct = (preds == labels)

    print(f"\n=== {name.upper()} RESULTS ===")
    print(f"Rows:     {len(df):,}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    plot_confusion_matrix(cm, title=f"{name.upper()} Confusion Matrix")
    plot_confidence_hist(confidence, correct, title=f"{name.upper()} Confidence: Correct vs Incorrect")


def main():
    device = get_device()
    print(f"Loading model from: {MODEL_DIR}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    val_df = load_split(f"{DATA_DIR}/val.csv")
    test_df = load_split(f"{DATA_DIR}/test.csv")

    evaluate_split("val", val_df, tokenizer, model, device)
    evaluate_split("test", test_df, tokenizer, model, device)


if __name__ == "__main__":
    main()