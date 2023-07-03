
import os
import st2_utils_acl
import pandas as pd
from datasets import Dataset 
from st2_utils_acl import load_df_from_tsv
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
import numpy as np
import wandb
from datasets import concatenate_datasets


#data params 
path_to_data = 'st2/processed_data/joined.tsv'

#output directory
output_dir_base = 'models/muppet-final/'
os.environ["WANDB_RUN_GROUP"] = "experiment-" + output_dir_base
output_dir = output_dir_base


#base model 
base_model = 'facebook/muppet-roberta-large'

#tokenizer_params
tokenizer_name = 'facebook/muppet-roberta-large'
max_length = 512

lr  = 3e-5

#load data from tsv
train_df = load_df_from_tsv(path_to_data, use_translations='en', convert_labels=True)
train_df = train_df[['id', 'text', 'label', 'lang']]
train_ds = Dataset.from_pandas(train_df)
train_ds = train_ds.shuffle()

print('train:', len(train_ds))


#tokenize 
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def tokenization(example): 
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=max_length) 

train_ds = train_ds.map(tokenization, batched=True)


def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    
    # we fill 1 to every class whose score is higher than some threshold
    # In this example, we choose that threshold = 0
    #with sigmoid function, threshold of 0 is equal to probability =0.5
    ret[:] = (logits[:] >= 0).astype(int)
    
    return ret


model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=14)

class MyTrainer(Trainer):
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss


class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: ")


training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    warmup_ratio=0.1, 
    optim = 'adamw_torch',


    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps = 1,
    save_strategy = 'epoch',
    save_total_limit = 2,


    weight_decay=0.01,

    report_to = ["wandb"],
    run_name = output_dir
)

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=predict_ds,
    callbacks=[PrinterCallback],
)

trainer.train()