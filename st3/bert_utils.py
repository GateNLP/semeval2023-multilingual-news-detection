"""Utility functions and classes specifically used to train transformer models."""

import torch
import numpy as np
import pandas as pd
from utils import load_dataset, upsample, upsamplev2, multibin

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from pytorch_lightning import LightningDataModule, LightningModule
from torch.optim import AdamW


class SemEvalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx]).float()

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# %%
class SemEvalDataModule(LightningDataModule):
    def __init__(
        self,
        pretrained_name: str,
        batch_size: int,
        max_seq_len: int,
        input_col: str,
        output_col: str,
        train_languages: list[str] = None,
        val_languages: list[str] = None,
        test_language: str = None,
        with_spans: bool = False,
        with_translations: bool = False,
        with_unlabelled: bool = True,
        upsample: str = None,
    ):
        super().__init__()
        self.train_languages = train_languages
        self.val_languages = val_languages
        self.test_language = test_language
        self.pretrained_name = pretrained_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.input_col = input_col
        self.output_col = output_col
        self.with_spans = with_spans
        self.with_translations = with_translations
        self.upsample = upsample
        self.with_unlabelled = with_unlabelled

    def prepare_data(self):
        """Tokenizes train, dev and test. Optionally does upsampling/class weighting."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name, use_fast=True)

        if self.train_languages is not None:
            train = load_dataset(
                self.train_languages,
                split="train",
                with_spans=self.with_spans,
                with_translations=self.with_translations,
                with_unlabelled=self.with_unlabelled,
            )

        if self.val_languages is not None:
            val = load_dataset(self.val_languages, split="dev", with_unlabelled=True)
            val_features = self.tokenizer(
                val[self.input_col].to_list(),
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            val_labels = multibin.transform(val[self.output_col].str.split(",").values)
            self.valset = SemEvalDataset(val_features, val_labels)

        # adds validation to train set if its not specified
        else:
            if self.train_languages is not None:
                train = pd.concat(
                    [
                        train,
                        load_dataset(
                            self.train_languages,
                            split="dev",
                            with_unlabelled=self.with_unlabelled,
                        ),
                    ]
                )
            self.valset = None

        if self.train_languages is not None:
            if self.upsample == "v1":
                train = upsample(train)
            elif self.upsample == "v2":
                train = upsamplev2(train)

            # train
            train_features = self.tokenizer(
                train[self.input_col].to_list(),
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            train_labels = multibin.transform(train[self.output_col].str.split(",").values)
            self.trainset = SemEvalDataset(train_features, train_labels)

            class_counts = train_labels.sum(axis=0)
            total = class_counts.sum()
            self.class_weights = [(total - count) / count if count > 0 else 1 for count in class_counts]
        else:
            self.trainset = None

        # test
        if self.test_language is not None:
            test = load_dataset(
                [self.test_language],
                split="test",
                with_spans=False,
                with_translations=False,
                with_unlabelled=False,
            )

            test_features = self.tokenizer(
                test[self.input_col].to_list(),
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )

            self.testset = SemEvalDataset(test_features)
        else:
            self.testset = None

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size)


# %%
class SemEvalTransformer(LightningModule):
    def __init__(
        self,
        pretrained_name: str,
        learning_rate: float,
        adam_epsilon: float,
        warmup_ratio: float,
        weight_decay: float,
        batch_size: int,
        max_seq_len: int,
        class_weights: list[float],
        classification_threshold: float,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_name,
            num_labels=len(multibin.classes_),
            problem_type="multi_label_classification",
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs[1]
        targets = batch["labels"]
        class_weights = self.hparams.class_weights
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights)
            class_weights = class_weights.to(self.device)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        loss = criterion(logits, targets)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        probs = torch.nn.Sigmoid()(logits)
        preds = torch.zeros(probs.shape)
        preds[torch.where(probs >= self.hparams.classification_threshold)] = 1
        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = []
        labels = []
        loss = []
        for output in outputs:
            preds.append(output["preds"].detach().cpu().numpy())
            labels.append(output["labels"].detach().cpu().numpy())
            loss.append(output["loss"].detach().cpu().item())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        loss = np.mean(loss)

        self.log("val_loss", loss, prog_bar=True)
        self.log("f1_micro", f1_score(labels, preds, average="micro"), prog_bar=True)
        self.log("acc", accuracy_score(labels, preds), prog_bar=True)
        self.log("f1_macro", f1_score(labels, preds, average="macro")),
        self.log("precision", precision_score(labels, preds, average="micro"))
        self.log("recall", recall_score(labels, preds, average="micro"))

        print(classification_report(labels, preds, target_names=multibin.classes_))

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=0.0,  #  weight_decay was already applied directly to each layer previously
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches * self.hparams.warmup_ratio,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
