
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SEED = 20260829
LABELS = ["MCD", "agreement"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-labelled",
        required=True,
        help="Control 600 human-labelled .xlsx file",
    )
    parser.add_argument(
        "--treatment-labelled",
        required=True,
        help="Treatment 600 human-labelled .xlsx file",
    )
    parser.add_argument(
        "--control-unlabelled",
        required=True,
        help="Full Control comment .xlsx file",
    )
    parser.add_argument(
        "--treatment-unlabelled",
        required=True,
        help="Full Treatment comment .csv file",
    )
    parser.add_argument("--output-dir", default="classifier_results")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def load_labelled(control_path, treatment_path, output_dir):
    control = pd.read_excel(control_path, sheet_name="control_sample_600")
    treatment = pd.read_excel(treatment_path, sheet_name="Audited_600")

    required = {"group", "comment_id", "comment_text", "human_MCD", "human_agreement"}

    for name, frame in [("Control", control), ("Treatment", treatment)]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} labelled file is missing required columns: {sorted(missing)}"
            )

    control_clean = pd.DataFrame(
        {
            "group": control["group"].astype(str).str.lower(),
            "comment_id": control["comment_id"].astype(str),
            "text": control["comment_text"].map(clean_text),
            "MCD": control["human_MCD"].astype(int),
            "agreement": control["human_agreement"].astype(int),
        }
    )

    treatment_clean = pd.DataFrame(
        {
            "group": treatment["group"].astype(str).str.lower(),
            "comment_id": treatment["comment_id"].astype(str),
            "text": treatment["comment_text"].map(clean_text),
            "MCD": treatment["human_MCD"].astype(int),
            "agreement": treatment["human_agreement"].astype(int),
        }
    )

    data = pd.concat([control_clean, treatment_clean], ignore_index=True)
    # 70% train, 15% validation, 15% held-out test,stratified by Treatment/Control group and the primary MCD label.
    strata = data["group"] + "_" + data["MCD"].astype(str)

    train_idx, temporary_idx = train_test_split(
        data.index,
        test_size=0.30,
        random_state=SEED,
        stratify=strata,
    )

    validation_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=strata.loc[temporary_idx],
    )

    data["split"] = ""
    data.loc[train_idx, "split"] = "train"
    data.loc[validation_idx, "split"] = "validation"
    data.loc[test_idx, "split"] = "test"

    data.to_csv(output_dir / "labelled_split.csv", index=False, encoding="utf-8-sig")

    return (
        data,
        train_idx.to_numpy(),
        validation_idx.to_numpy(),
        test_idx.to_numpy(),
    )


def load_unlabelled(control_path, treatment_path):
    treatment = pd.read_csv(treatment_path, encoding="utf-8-sig")
    control = pd.read_excel(control_path, sheet_name="control_comments")

    if "comment_text" not in treatment.columns:
        raise ValueError("Treatment full comment file must contain 'comment_text'.")
    if "comment_text" not in control.columns:
        raise ValueError("Control full comment file must contain 'comment_text'.")

    treatment["model_text"] = treatment["comment_text"].map(clean_text)
    control["model_text"] = control["comment_text"].map(clean_text)

    return treatment, control


def choose_threshold(y_true, probability):
    candidates = np.linspace(0.05, 0.95, 181)
    scores = []

    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        f1 = precision_recall_fscore_support(
            y_true,
            prediction,
            average="binary",
            zero_division=0,
        )[2]
        scores.append(f1)

    return float(candidates[int(np.argmax(scores))])


def evaluate(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        prediction,
        average="binary",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "predicted_positives": int(np.sum(prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_predictions(
    treatment,
    control,
    treatment_probability,
    control_probability,
    report,
    output_dir,
):
    for frame, probability, group_name in [
        (treatment, treatment_probability, "treatment"),
        (control, control_probability, "control"),
    ]:
        result = frame.drop(columns="model_text").copy()

        for j, label in enumerate(LABELS):
            threshold = report[label]["test"]["threshold"]
            prob = probability[:, j]

            result[f"BERT_{label}_prob"] = prob
            result[f"BERT_{label}_pred"] = (
                prob >= threshold
            ).astype(int)

        result.to_csv(
            output_dir / f"pred_{group_name}_BERT.csv",
            index=False,
            encoding="utf-8-sig",
        )


def train_bert(
    data,
    train_idx,
    validation_idx,
    test_idx,
    treatment,
    control,
    output_dir,
    epochs,
    max_length,
):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    model_id = "google-bert/bert-base-uncased"

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        problem_type="multi_label_classification",
    ).to(device)

    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = list(texts)
            self.labels = (
                None
                if labels is None
                else np.asarray(labels, dtype=np.float32)
            )

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            item = tokenizer(
                self.texts[index],
                truncation=True,
                max_length=max_length,
            )
            if self.labels is not None:
                item["labels"] = self.labels[index]
            return item

    def collate(batch):
        labels = None

        if "labels" in batch[0]:
            labels = torch.tensor(
                np.stack([item.pop("labels") for item in batch]),
                dtype=torch.float32,
            )

        padded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

        if labels is not None:
            padded["labels"] = labels

        return padded

    train_loader = DataLoader(
        TextDataset(
            data.loc[train_idx, "text"],
            data.loc[train_idx, LABELS],
        ),
        batch_size=8,
        shuffle=True,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    y_train = data.loc[train_idx, LABELS].to_numpy()
    positive = y_train.sum(axis=0)

    positive_weight = torch.tensor(
        (len(y_train) - positive) / np.maximum(positive, 1),
        dtype=torch.float32,
        device=device,
    )

    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            logits = model(**batch).logits
            loss = loss_function(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader),
            }
        )

    def infer(texts, batch_size=24):
        texts = list(texts)

        # Sorting by text length only makes inference more efficient.
        # Predictions are returned to their original order afterwards.
        order = np.argsort(
            [len(text) for text in texts],
            kind="stable",
        )

        loader = DataLoader(
            TextDataset([texts[i] for i in order]),
            batch_size=batch_size,
            collate_fn=collate,
        )

        chunks = []
        model.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }
                probability = torch.sigmoid(
                    model(**batch).logits
                )
                chunks.append(
                    probability.cpu().numpy()
                )

        sorted_probability = np.concatenate(chunks)

        probability = np.empty_like(sorted_probability)
        probability[order] = sorted_probability

        return probability

    validation_probability = infer(
        data.loc[validation_idx, "text"]
    )
    test_probability = infer(
        data.loc[test_idx, "text"]
    )

    report = {
        "_meta": {
            "model_id": model_id,
            "seed": SEED,
            "epochs": epochs,
            "max_length": max_length,
            "device": str(device),
            "history": history,
        }
    }

    for j, label in enumerate(LABELS):
        validation_y = data.loc[
            validation_idx, label
        ].to_numpy()

        test_y = data.loc[
            test_idx, label
        ].to_numpy()

        threshold = choose_threshold(
            validation_y,
            validation_probability[:, j],
        )

        report[label] = {
            "validation": evaluate(
                validation_y,
                validation_probability[:, j],
                threshold,
            ),
            "test": evaluate(
                test_y,
                test_probability[:, j],
                threshold,
            ),
        }

    model.save_pretrained(
        output_dir / "model_BERT",
        safe_serialization=False,
    )
    tokenizer.save_pretrained(
        output_dir / "model_BERT"
    )

    treatment_probability = infer(
        treatment["model_text"]
    )
    control_probability = infer(
        control["model_text"]
    )

    save_predictions(
        treatment,
        control,
        treatment_probability,
        control_probability,
        report,
        output_dir,
    )

    del model
    gc.collect()

    return report


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(SEED)
    np.random.seed(SEED)

    data, train_idx, validation_idx, test_idx = load_labelled(
        args.control_labelled,
        args.treatment_labelled,
        output_dir,
    )

    treatment, control = load_unlabelled(
        args.control_unlabelled,
        args.treatment_unlabelled,
    )

    report = train_bert(
        data,
        train_idx,
        validation_idx,
        test_idx,
        treatment,
        control,
        output_dir,
        args.epochs,
        args.max_length,
    )

    (output_dir / "metrics_BERT.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
