import argparse
import glob

import pandas as pd
import numpy as np
import pathlib

langs = ['ge', 'en', 'fr', 'it', 'po', 'ru']

def _strip_url(text):
    """Remove the article URL on line 2 of the data."""
    lines = text.split('\n')

    new_lines = [lines[0], ''] + lines[2:]

    return '\n'.join(new_lines)

def create():
    satire_files = glob.glob('../data/ext_satire/*.txt')
    satire_articles = {}

    for file in satire_files:
        article_id = file[16:-4]
        try:
            with open(file, 'r', encoding='utf8') as f:
                file_contents = f.read()
        except UnicodeDecodeError:
            with open(file, 'r', encoding='cp1252') as f:
                file_contents = f.read()

        if article_id in satire_articles:
            print("DUPLICATED", article_id)

        satire_articles[article_id] = file_contents

    df = pd.DataFrame.from_dict(satire_articles, orient='index', columns=['text'])
    df['label'] = 'satire'
    df['language'] = 'en'
    df['article'] = df.index.map(lambda i: f"12345{i}")
    df['text'] = df['text'].apply(_strip_url)
    df['lang_label'] = df.apply(lambda r: f"{r['language']}_{r['label']}", axis=1)
    df[['article', 'text', 'language', 'label', 'lang_label']].to_csv('../data/ext_satire/external_satire.tsv', sep = '\t')

def split():
    df = pd.read_csv('../data/ext_satire/external_satire.tsv', sep='\t', index_col=0)
    df = df.sample(frac=1, random_state=2023)
    print(len(df), "external satire loaded")

    chunked = np.array_split(df, len(langs))

    pathlib.Path('../data/ext_satire/to_translate').mkdir(exist_ok=True)

    for i, chunk in enumerate(chunked):
        print(langs[i], len(chunk))
        chunk.to_excel(f'../data/ext_satire/to_translate/{langs[i]}.xlsx')

    print("Now translate each split and put in data/ext_satire/translated/")


def combine():
    lang_dfs = []
    for lang in langs:
        lang_df = pd.read_excel(f'../data/ext_satire/translated/{lang}.xlsx',
                                names=['article', 'text', 'language', 'label'],
                                index_col=0)
        lang_df['language'] = lang
        lang_df['label'] = 'satire'
        lang_dfs.append(lang_df)

    all_langs = pd.concat(lang_dfs)
    all_langs['lang_label'] = all_langs.apply(lambda r: f"{r['language']}_{r['label']}",
                                              axis=1)

    all_langs.to_csv('../data/ext_satire/external_satire_translated.tsv', sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['create', 'split', 'combine'],
                        help='create: merge satire txt files into a tsv. split: '
                             'create a split for each target language. combine: '
                             'recombine translated splits.')
    args = parser.parse_args()

    if args.task == 'create':
        create()
    elif args.task == 'split':
        split()
    elif args.task == 'combine':
        combine()
