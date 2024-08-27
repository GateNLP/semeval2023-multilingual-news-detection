import pandas as pd
import pathlib

if __name__ == "__main__":
    data_root = pathlib.Path(__file__).parent.parent / 'data'
    language_dirs = [x for x in data_root.iterdir() if x.is_dir() and len(x.name) == 2]
    print("Languages:", [lang.name for lang in language_dirs])

    train_set = []
    dev_set = []

    for lang in language_dirs:
        print("Processing", lang.name)
        train_label_file = lang / f'train-labels-subtask-1.txt'

        if train_label_file.exists():
            train_labels = pd.read_csv(lang / f'train-labels-subtask-1.txt', skiprows=0, 
                                    encoding = "utf-8", names=['article', 'label'], delimiter='\t')
            train_labels['article'] = train_labels['article'].astype("string")
            train_labels['label'] = train_labels['label'].astype("string")

            train_articles_dir = lang / f'train-articles-subtask-1'
            n = 0
            for file in train_articles_dir.glob('*.txt'):
                n += 1
                idx = file.name[7:-4]
                file_content = file.read_text()
                gold_label = train_labels.loc[train_labels['article'] == idx].label.values[0]
                train_set.append([idx, file_content, lang.name, gold_label])
            print("\tFound", n, "train")
        else:
            print("\tNo train data found for", lang.name)

        dev_articles_dir = lang / f'dev-articles-subtask-1'
        if dev_articles_dir.exists():
            dev_articles_dir = lang / f'dev-articles-subtask-1'
            n = 0
            for file in dev_articles_dir.glob('*.txt'):
                n += 1
                idx = file.name[7:-4]
                file_content = file.read_text()
                dev_set.append([idx, file_content, lang.name])
            print("\tFound", n, "dev")
        else:
            print("\tNo dev data found for", lang.name)
    
    train_df = pd.DataFrame(train_set, columns=['article', 'text', 'language', 'label'])
    dev_df = pd.DataFrame(dev_set, columns=['article', 'text', 'language'])


    joined_data_root = data_root / 'st1_joined_data'
    joined_data_root.mkdir(exist_ok=True)

    train_df.to_csv(joined_data_root / 'training.tsv', sep = '\t')
    dev_df.to_csv(joined_data_root / 'dev.tsv', sep = '\t')