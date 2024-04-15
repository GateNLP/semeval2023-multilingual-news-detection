#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:38:14 2024

@author: lesya
"""

# Libraries
import csv
import re
import numpy as np
import pandas as pd
import random
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertModel, CamembertTokenizer
from sklearn import preprocessing
from nltk.tokenize import sent_tokenize
import re
from sklearn.metrics import f1_score, classification_report


if torch.cuda.is_available():      
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

batch_size = 32
EPOCHS = 25
LEARNING_RATE = 1e-5
n_class=3
loss_fn = nn.CrossEntropyLoss()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")



training_set=pd.read_csv("./data/cleaned_original_train.tsv", skiprows=1,
                          encoding = "utf-8", names=['article', 'text', 'language', 'label', 'isextrnl', 'istrsl'], delimiter='\t')
# compile satire dataset to upsample from 
satire_en=pd.read_csv("./data/satire_external_en.tsv", skiprows=1,
                          encoding = "utf-8", names=['article', 'text', 'language', 'label', 'isextrnl', 'istrsl'], delimiter='\t')

dev_set=pd.read_csv("./data/cleaned_original_dev.tsv", skiprows=1,
                           encoding = "utf-8", names=['article', 'text', 'language', 'label', 'isextrnl', 'istrsl'], delimiter='\t')

# split the dev set into 3 stratified folds to use 1 for inference
fold1, fold2=train_test_split(dev_set, test_size=0.3,  stratify=dev_set[['label', 'language']])
fold2, fold3=train_test_split(fold1, test_size=0.5,  stratify=dev_set[['label', 'language']])
dev1=pd.concat([fold1, fold2])
dev2=pd.concat([fold1, fold3])
dev3=pd.concat([fold2, fold3])
devs=set(dev1, dev2, dev3)
val_sets=set(fold1, fold2, fold3)


def resplit_data(train, dev):    
    training_set=pd.concat([train, dev])
    training_set = training_set.sample(frac=1).reset_index(drop=True)
    # use this if you do not do the stratified held-out val test split
    # training_set, val_set=train_test_split(training_set, test_size=0.1,  stratify=training_set[['label', 'language']])
    return training_set

# change from common to ru only below
def upsample(lang):     
    data_upsampled=training_set[training_set["language"]==lang]
    if lang!="en":
        while True:
            for i, row in data_upsampled[data_upsampled["label"]=="satire"].iterrows():
                if len(data_upsampled[data_upsampled["label"]=="satire"])<len(data_upsampled[data_upsampled["label"]=="opinion"]):
                    data_upsampled=data_upsampled.append(row)
            if len(data_upsampled[data_upsampled["label"]=="satire"])>=len(data_upsampled[data_upsampled["label"]=="opinion"]):
                break
    else:
        for i, row in satire_en.iterrows():
            data_upsampled=data_upsampled.append(row)
     
    while True:
        for i, row in data_upsampled[data_upsampled["label"]=="reporting"].iterrows():
            if len(data_upsampled[data_upsampled["label"]=="reporting"])<len(data_upsampled[data_upsampled["label"]=="opinion"]):
                data_upsampled=data_upsampled.append(row)
        if len(data_upsampled[data_upsampled["label"]=="reporting"])>=len(data_upsampled[data_upsampled["label"]=="opinion"]):
            break
    return data_upsampled





def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    text=text.strip()
    sents=tokenizer.tokenize(text)
    # print(len(text))
    # print(len(sents))
    start=""
    end=""
    if len(sents)>512:
        # print("words "+str(len(sents)))
        m=re.split('(?<=[\.\?\!])\s*', text)
        # start eppending sentences to the end and start
        for count,sent in enumerate(m):
            if len(tokenizer.tokenize(start+" "+end))<512:
                start=start+" "+sent
                if len(m)-count-1<count:
                    end=end+str(m[len(m)-count-1:len(m)-count])
        text=start+" "+end
    return text

def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # create empty lists to store outputs
    input_ids = []
    attention_masks = []
    #for every sentence...
    for sent in data:
        sent=str(sent)
        # 'encode_plus will':
        # (1) Tokenize the sentence
        # (2) Add the `[CLS]` and `[SEP]` token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map tokens to their IDs
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text = text_preprocessing(sent),   #preprocess sentence
            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`
            max_length= 512  ,             #Max length to truncate/pad
            pad_to_max_length = True,          #pad sentence to max length
            return_attention_mask= True        #Return attention mask
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    #convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids,attention_masks


# translate labels into numeric values(needed to create a torch tensor)
def encode(train_set, val_set):
    le = preprocessing.LabelEncoder()
    le.fit(train_set.label.values)
    train_set.label=le.transform(all_upsampled.label)
    val_set.label=le.transform(val_set.label)
    
    # Run function 'preprocessing_for_bert' on the train set and validation set
    print('Tokenizing data...')
    # fold_training = fold_training.sample(frac=1).reset_index(drop=True)
    X_train= train_set.text.values
    y_train= train_set.label.values

    X_val_en= val_set[val_set["language"]=="en"].text.values
    y_val_en= val_set[val_set["language"]=="en"].label.values
    X_val_fr= val_set[val_set["language"]=="fr"].text.values
    y_val_fr= val_set[val_set["language"]=="fr"].label.values
    X_val_ge= val_set[val_set["language"]=="ge"].text.values
    y_val_ge= val_set[val_set["language"]=="ge"].label.values
    X_val_it= val_set[val_set["language"]=="it"].text.values
    y_val_it= val_set[val_set["language"]=="it"].label.values
    X_val_po= val_set[val_set["language"]=="po"].text.values
    y_val_po= val_set[val_set["language"]=="po"].label.values
    X_val_ru= val_set[val_set["language"]=="ru"].text.values
    y_val_ru= val_set[val_set["language"]=="ru"].label.values
    
    
    
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs_en, val_masks_en = preprocessing_for_bert(X_val_en)
    val_inputs_fr, val_masks_fr = preprocessing_for_bert(X_val_fr)
    val_inputs_it, val_masks_it = preprocessing_for_bert(X_val_it)
    val_inputs_ge, val_masks_ge = preprocessing_for_bert(X_val_ge)
    val_inputs_po, val_masks_po = preprocessing_for_bert(X_val_po)
    val_inputs_ru, val_masks_ru = preprocessing_for_bert(X_val_ru)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels_en = torch.tensor(y_val_en)
    val_labels_fr = torch.tensor(y_val_fr)
    val_labels_it = torch.tensor(y_val_it)
    val_labels_ge = torch.tensor(y_val_ge)
    val_labels_po = torch.tensor(y_val_po)
    val_labels_ru = torch.tensor(y_val_ru)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs,train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Create the DataLoader for our validation set
    val_data_en = TensorDataset(val_inputs_en, val_masks_en, val_labels_en)
    val_sampler_en = RandomSampler(val_data_en)
    val_dataloader_en = DataLoader(val_data_en, sampler=val_sampler_en, batch_size=batch_size)
    
    val_data_fr = TensorDataset(val_inputs_fr, val_masks_fr, val_labels_fr)
    val_sampler_fr = RandomSampler(val_data_fr)
    val_dataloader_fr = DataLoader(val_data_fr, sampler=val_sampler_fr, batch_size=batch_size)
    
    val_data_ge = TensorDataset(val_inputs_ge, val_masks_ge, val_labels_ge)
    val_sampler_ge = RandomSampler(val_data_ge)
    val_dataloader_ge = DataLoader(val_data_ge, sampler=val_sampler_ge, batch_size=batch_size)
    
    val_data_it = TensorDataset(val_inputs_it, val_masks_it, val_labels_it)
    val_sampler_it = RandomSampler(val_data_it)
    val_dataloader_it = DataLoader(val_data_it, sampler=val_sampler_it, batch_size=batch_size)
    
    val_data_po = TensorDataset(val_inputs_po, val_masks_po, val_labels_po)
    val_sampler_po = RandomSampler(val_data_po)
    val_dataloader_po = DataLoader(val_data_po, sampler=val_sampler_po, batch_size=batch_size)
    
    val_data_ru = TensorDataset(val_inputs_ru, val_masks_ru, val_labels_ru)
    val_sampler_ru = RandomSampler(val_data_ru)
    val_dataloader_ru = DataLoader(val_data_ru, sampler=val_sampler_ru, batch_size=batch_size)
    
    return train_dataloader, val_dataloader_en, val_dataloader_fr, val_dataloader_ge, val_dataloader_it, val_dataloader_po, val_dataloader_ru



class BertClassifier(nn.Module):
    """
        Bert Model for classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param   bert: a BertModel object
        @param   classifier: a torch.nn.Module classifier
        @param   freeze_bert (bool): Set `False` to fine_tune the Bert model
        """
        super(BertClassifier,self).__init__()
        # Specify hidden size of Bert, hidden size of our classifier, and number of labels
        D_in, H,D_out = 768,50,n_class
       
        # self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        # self.bert = BertModel.from_pretrained("bert-base-cased")
        self.bert = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        # self.bert = BertModel.from_pretrained("dbmdz/bert-base-italian-cased")
        # self.bert = BertModel.from_pretrained("camembert/camembert-base") 
        # self.bert = CamembertModel.from_pretrained("camembert-base")
       
        self.classifier = nn.Sequential(
                            nn.Linear(D_in, H),
                            nn.ReLU(),
                            nn.Linear(H, D_out))
        # self.sigmoid = nn.Sigmoid()
        self.softmax=nn.Softmax()
        # Freeze the Bert Model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
   
    def forward(self,input_ids,attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        outputs = self.bert(input_ids=input_ids,
                           attention_mask = attention_mask)
       
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:,0,:]
       
        # Feed input to classifier to compute logits
        logit = self.classifier(last_hidden_state_cls)
       
        # logit = self.softmax(logit)
       
        return logit
   



def initialize_model(train_dataloader, epochs=EPOCHS):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
   
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)
   
    bert_classifier.to(device)
   
    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                     lr=1e-5, #Default learning rate
                     eps=1e-8 #Default epsilon value
                     )
   
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
   
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0, # Default value
                                              num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler    



# Specify loss function
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCEWithLogitsLoss()

import torch.nn.functional as F
def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
   
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    # probs=all_logits.sigmoid().cpu().numpy()
    return probs


       
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
       
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


def create_test(test_dataframe):
    # Run `preprocessing_for_bert` on the test set
    test_inputs, test_masks = preprocessing_for_bert(test_dataframe.text)
    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    print('The test set is ready')
    return test_dataloader

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)







def train(model, train_dataloader, val_dataloader_en, val_dataloader_fr, val_dataloader_ge, val_dataloader_it, val_dataloader_po, val_dataloader_ru, epochs=30, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    best_val_en=0
    best_val_fr=0
    best_val_it=0
    best_val_ge=0
    best_val_ru=0
    best_val_po=0
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            print(torch.argmax(logits, dim=1).flatten())
            print(b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                print(torch.argmax(logits, dim=1).flatten())
                print(b_labels)
                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        
        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss_en, val_accuracy_en = evaluate(model, val_dataloader_en)
            if val_accuracy_en>best_val_en:
                torch.save(bert_classifier,"./data/bmult_best_en")
                best_val_en=val_accuracy_en
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch 
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss_en:^10.6f} | {val_accuracy_en:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            val_loss_fr, val_accuracy_fr = evaluate(model, val_dataloader_fr)
            if val_accuracy_fr>best_val_fr:
                torch.save(bert_classifier,"./data/bmult_best_fr")
                best_val_fr=val_accuracy_fr
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss_fr:^10.6f} | {val_accuracy_fr:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            val_loss_it, val_accuracy_it = evaluate(model, val_dataloader_it)
            if val_accuracy_it>best_val_it:
                torch.save(bert_classifier,"./data/bmult_best_it")
                best_val_it=val_accuracy_it
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss_it:^10.6f} | {val_accuracy_it:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            val_loss_ge, val_accuracy_ge = evaluate(model, val_dataloader_ge)
            if val_accuracy_ge>best_val_ge:
                torch.save(bert_classifier,"./data/bmult_best_ge")
                best_val_ge=val_accuracy_ge
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch           
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss_ge:^10.6f} | {val_accuracy_ge:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            val_loss_po, val_accuracy_po = evaluate(model, val_dataloader_po)
            if val_accuracy_po>best_val_po:
                torch.save(bert_classifier,"./data/bmult_best_po")
                best_val_po=val_accuracy_po
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch 
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss_po:^10.6f} | {val_accuracy_po:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            val_loss_ru, val_accuracy_ru = evaluate(model, val_dataloader_ru)
            if val_accuracy_ru>best_val_ru:
                torch.save(bert_classifier,"./data/bmult_best_ru")
                best_val_ru=val_accuracy_ru
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss_ru:^10.6f} | {val_accuracy_ru:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
   
    print("Training complete!")
   


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    predictions=[]
    gold_lab=[]
    report=[]
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
       

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits,b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
        predictions.append(preds)
        gold_lab.append(b_labels)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy



def text_preprocessing_test(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    text=text.strip()
    sents=tokenizer.tokenize(text)
    print(len(text))
    print(len(sents))
    
    if len(sents)>512:
        # print("words "+str(len(sents)))
        m=re.split('(?<=[\.\?\!])\s*', text)
        start=""
        end=""
        # start eppending sentences to the end and start
        for count,sent in enumerate(m):
            if count<=10 and len(tokenizer.tokenize(start+" "+end))<512:
                start=start+" "+sent
                if len(m)-count-1<count:
                    end=end+str(m[len(m)-count-1:len(m)-count])
            if count>10 and len(tokenizer.tokenize(start+" "+end))<512:
                start=start+" "+sent
    return text

def preprocessing_for_bert_test(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # create empty lists to store outputs
    input_ids = []
    attention_masks = []
   
    #for every sentence...
   
    for sent in data:
        sent=str(sent)
        # 'encode_plus will':
        # (1) Tokenize the sentence
        # (2) Add the `[CLS]` and `[SEP]` token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map tokens to their IDs
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text = text_preprocessing(sent),   #preprocess sentence
            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`
            max_length= 512  ,             #Max length to truncate/pad
            pad_to_max_length = True,          #pad sentence to max length
            return_attention_mask= True        #Return attention mask
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
       
    #convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
   
    return input_ids,attention_masks




set_seed()    # Set seed for reproducibility
i=0
for fold in devs:
    training_set=resplit_data(training_set, fold)
    df1=upsample("en")
    df2=upsample("po")
    df3=upsample("ru")
    df4=upsample("ge")
    df5=upsample("it")
    df6=upsample("fr")
    all_upsampled=pd.concat([df1,df2, df3, df4, df5, df6])
    all_upsampled = all_upsampled.sample(frac=1).reset_index(drop=True)
        
    
    train_dataloader, val_dataloader_en,val_dataloader_fr, val_dataloader_ge, val_dataloader_it, val_dataloader_po, val_dataloader_ru =encode(train_set=all_upsampled, val_set=val_sets[i])
    bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)
    train(bert_classifier, train_dataloader, val_dataloader_fr, val_dataloader_ge, val_dataloader_it, val_dataloader_po, val_dataloader_ru, epochs=EPOCHS, evaluation=True)
    i=i+1
    



# Test after the training is finished
test_set=pd.read_csv("./data/test_set.tsv", skiprows=1,
                          encoding = "utf-8", names=['article', 'text', 'language', 'label'], delimiter='\t')
le = preprocessing.LabelEncoder()
le.fit(training_set.label.values)
for lang in ["en", "fr", "it", "ge", "po", "ru"]:
    test_lang=test_set[test_set["language"]==lang]
    bert_classifier=torch.load("./joined_data/bmult_best_"+lang)
    probs = bert_predict(bert_classifier, create_test(test_lang))
    probs=probs.argmax(1)
    probs=le.inverse_transform(probs)
    csv_file=open('./data/predictions_'+lang+'.txt', 'w')
    writer = csv.writer(csv_file, delimiter='\t')
    i=0
    for ind, row in test_lang.iterrows():
        writer.writerow([row["article"],probs[i]])
        i=i+1 