import pathlib

import dataclasses
from functools import partial
from sklearn.metrics import f1_score, confusion_matrix
import torch
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AdapterConfig, set_seed, HfArgumentParser, AutoConfig, AutoModelWithHeads, \
    TrainingArguments, TrainerCallback, EvalPrediction, AdapterTrainer, PfeifferConfig
import wandb
from args import DataArguments, ModelArguments, MyTrainingArguments
import numpy as np

default_adapter_config = {
    "mh_adapter": True,
    "output_adapter": True,
    "non_linearity": "relu"
}


def encode_batch(tokenizer, batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], truncation=True, padding="max_length")


def load_df(path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1,
                       encoding="utf-8", names=['article', 'text', 'language', 'label', 'lang_label'], delimiter='\t')


def make_dataset(data_args: DataArguments, tokenizer: AutoTokenizer, df, split=False, test_size=0.1) -> Dataset:
    dataset = Dataset.from_pandas(df).class_encode_column('label').class_encode_column('lang_label')
    dataset = dataset.map(partial(encode_batch, tokenizer), batched=True)
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", 'lang_label'])

    if split:
        dataset = dataset.train_test_split(test_size=test_size, seed=2023, stratify_by_column=data_args.stratify_field)

    return dataset


def prepare_data(data_args: DataArguments, tokenizer: AutoTokenizer) -> DatasetDict:
    train_val_df = load_df('../../data/st1_joined_data/training.tsv')
    test_df = load_df('../../data/st1_joined_data/dev.tsv')

    if data_args.hold_out_lang is not None:
        should_holdout = train_val_df['language'] == data_args.hold_out_lang
        holdout_data = train_val_df[should_holdout]
        train_val_df = train_val_df[~should_holdout]
        print(f"Holding out {len(holdout_data)} examples from training")
        holdout_dataset = make_dataset(data_args, tokenizer, holdout_data)
    else:
        holdout_dataset = None

    if data_args.merge_and_resplit:
        merged_df = pd.concat([train_val_df, test_df])
        train_valtest = make_dataset(data_args, tokenizer, merged_df, split=True, test_size=0.2)
        val_test = train_valtest['test'].train_test_split(test_size=0.5, seed=2023, stratify_by_column=data_args.stratify_field)

        train_dataset = train_valtest['train']
        val_dataset = val_test['train']
        test_dataset = val_test['test']
    else:
        train_val_dataset = make_dataset(data_args, tokenizer, train_val_df, split=True)

        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = make_dataset(data_args, tokenizer, test_df, split=False)

    if holdout_dataset is not None:
        test_dataset = concatenate_datasets([test_dataset, holdout_dataset])

    if data_args.external_satire:
        external_satire_df = load_df('../../data/ext_satire/external_satire.tsv')
        external_satire_dataset = make_dataset(data_args, tokenizer, external_satire_df, split=True, test_size=0.1)
        # we have to cast the external satire dataset explicitly because its Label will have different features
        external_satire_dataset = external_satire_dataset.cast(train_dataset.features)
        train_dataset = concatenate_datasets([train_dataset, external_satire_dataset['train']])
        val_dataset = concatenate_datasets([val_dataset, external_satire_dataset['test']])

    if data_args.external_satire_translated:
        external_satire_translated_df = load_df('../../data/ext_satire/external_satire_translated.tsv')
        external_satire_translated_dataset = make_dataset(data_args, tokenizer, external_satire_translated_df, split=True, test_size=0.1)
        external_satire_translated_dataset = external_satire_translated_dataset.cast(train_dataset.features)
        train_dataset = concatenate_datasets([train_dataset, external_satire_translated_dataset['train']])
        val_dataset = concatenate_datasets([val_dataset, external_satire_translated_dataset['test']])

    dataset = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

    return dataset


def save_dataset(dataset: DatasetDict, model_name: str):
    pathlib.Path(f'./training_output/{model_name}/data').mkdir(parents=True, exist_ok=True)
    dataset_artifact = wandb.Artifact("dataset", type="preprocessed_data")
    for split in dataset.keys():
        dataset[split].to_csv(f'./training_output/{model_name}/data/{split}.csv', columns=['article', 'text', 'language', 'labels'])
    dataset_artifact.add_dir(f'./training_output/{model_name}/data')
    wandb.log_artifact(dataset_artifact)


class AdapterDropTrainerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        skip_layers = list(range(np.random.randint(0, 11)))
        kwargs['model'].set_active_adapters("myadapter", skip_layers=skip_layers)

    def on_evaluate(self, args, state, control, **kwargs):
        # Deactivate skipping layers during evaluation (otherwise it would use the
        # previous randomly chosen skip_layers and thus yield results not comparable
        # across different epochs)
        kwargs['model'].set_active_adapters("myadapter", skip_layers=None)


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).mean()
    f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro')
    cmat = confusion_matrix(p.label_ids, preds).tolist()

    return {"acc": acc, "f1": f1, "cmat": cmat}


def make_adapter_config(model_args: ModelArguments):
    if model_args.adapter_config_type == 'base':
        return AdapterConfig(reduction_factor=model_args.adapter_reduction_factor, **default_adapter_config)
    elif model_args.adapter_config_type == 'pfeiffer':
        return PfeifferConfig(reduction_factor=model_args.adapter_reduction_factor)
    else:
        raise ValueError('Invalid adapter config type. Must be "base" or "pfeiffer"')


def main(data_args: DataArguments, model_args: ModelArguments, train_args: MyTrainingArguments):
    tokenizer = AutoTokenizer.from_pretrained(model_args.name)

    dataset = prepare_data(data_args, tokenizer)

    config = AutoConfig.from_pretrained(model_args.name, num_labels=3)
    model = AutoModelWithHeads.from_pretrained(model_args.name, config=config)

    adapter_config = make_adapter_config(model_args)

    model.add_adapter("myadapter", config=adapter_config)
    model.add_classification_head("myadapter", num_labels=3)
    model.train_adapter("myadapter")

    save_dataset(dataset, wandb.run.name)

    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=10,
        output_dir=f"./training_output/{wandb.run.name}",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        report_to="wandb",
        **dataclasses.asdict(train_args)
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=compute_metrics,
    )

    if model_args.adapter_dropping:
        trainer.add_callback(AdapterDropTrainerCallback())

    trainer.train()

    print("\n\n========= TESTING ======== \n")

    trainer.model.save_adapter(f"./training_output/{wandb.run.name}/best_adapter", "myadapter")
    adapter_artifact = wandb.Artifact("best_adapter", type="model")
    adapter_artifact.add_dir(f"./training_output/{wandb.run.name}/best_adapter")
    wandb.log_artifact(adapter_artifact)

    test_out = trainer.predict(dataset['test'])
    names_arr = np.array(dataset['test'].features['labels'].names)
    wandb.log({k.replace("_", "/", 1): v for k, v in test_out.metrics.items()})

    print(test_out.metrics)
    test_preds = np.argmax(test_out.predictions, axis=1)

    cmat = wandb.plot.confusion_matrix(y_true=test_out.label_ids, preds=test_preds, class_names=names_arr,
                                       title="All Languages")
    wandb.log({"cmat_all": cmat})

    test_dataset_df = dataset['test'].to_pandas()[['article', 'text', 'language', 'labels']]
    test_dataset_df['pred'] = test_preds

    for lang in test_dataset_df['language'].unique():
        examples = test_dataset_df[test_dataset_df['language'] == lang]
        lang_labels = examples['labels'].to_numpy()
        lang_preds = examples['pred'].to_numpy()

        f1 = f1_score(lang_labels, lang_preds, average='macro')
        wandb.log({f"test/lang_f1/{lang}": f1})

        lang_cmat = wandb.plot.confusion_matrix(y_true=lang_labels, preds=lang_preds, class_names=names_arr,
                                                title=f"Language {lang}")
        wandb.log({f"test/lang_cmat/{lang}": lang_cmat})

    # Delete the massive adapter config object
    wandb.config.update({'adapters': None}, allow_val_change=True)


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, MyTrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    set_seed(2024)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    wandb.login()
    wandb.init(project="sem233-st1-adapters2", entity="gate-vigilant")
    wandb.define_metric("eval/f1", summary="max")

    data_args.wandb_push()
    model_args.wandb_push()
    train_args.wandb_push()

    main(data_args, model_args, train_args)