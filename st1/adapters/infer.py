import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig, BertAdapterModel, \
    TextClassificationPipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
from glob import glob
import os

LANGS = ['en', 'es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'ru']
ROOT = '../../data/'
label_decoder = {
    "LABEL_0": "opinion",
    "LABEL_1": "reporting",
    "LABEL_2": "satire"
}


def main(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    config = AutoConfig.from_pretrained("bert-base-multilingual-cased", num_labels=3)
    model = BertAdapterModel.from_pretrained("bert-base-multilingual-cased",
                                             config=config)

    model.load_adapter(f"training_output/{model_name}/best_adapter", set_active=True)
    model.to('cuda')

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

    lang_dfs = []
    for lang in LANGS:
        files = []
        test_articles = ROOT + '/' + lang + '/test-articles-subtask-1/*.txt'
        articles = glob(test_articles)

        for article in articles:
            idx = os.path.basename(article)[7:-4]
            with open(article, 'r') as f:
                files.append((idx, lang, f.read()))

        lang_dfs.append(pd.DataFrame(files, columns=['id', 'lang', 'text']))
    df = pd.concat(lang_dfs)

    print(df.lang.value_counts())

    dataset = Dataset.from_pandas(df)
    labels = []
    for out in classifier(KeyDataset(dataset, "text"), padding='max_length',
                          truncation=True):
        labels.append(out['label'])

    df['labels'] = labels
    df['labels'] = df['labels'].replace(label_decoder)

    print(df['labels'].value_counts())

    for lang in LANGS:
        lang_df = df[df['lang'] == lang]
        lang_df[['id', 'labels']].to_csv(f'preds/{lang}.tsv', sep='\t', index=False,
                                         header=False)

    df[['id', 'lang', 'labels']].to_csv(f'preds/ALL.tsv', sep='\t', index=False,
                                        header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model name to make predictions on")
    args = parser.parse_args()

    Path('./preds').mkdir(exist_ok=True)

    main(args.model_name)
