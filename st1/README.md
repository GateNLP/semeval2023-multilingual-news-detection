# Subtask 1

## External Satire Data

Download the articles from [Goldbeck et. al's Fake News Dataset](https://github.com/jgolbeck/fakenews). Extract the satire article text files into `./data/ext_satire/articles/`.

Use the `process_external_satire.py` script to convert this data into the same format as the task training data:

```shell
python3 process_external_satire.py create
```

To translate the data into splits for each language, use the script again with the `split` arg to produce one Excel sheet per language. We used Google Translate's document translation feature. Then use it with the `combine` arg to merge them back into a single TSV.

## Non-Adapter Models
TODO

## Adapter Models

Since we use [`adapter-transformers`](https://github.com/adapter-hub/adapter-transformers), which conflicts with the regular `transformers` package, this requires a [separate Python environment](st1/adapters/env.yml).

To train the model used in the paper, run:

```shell
python3 train.py --adapter_reduction_factor 8 --external_satire
```

## Ensemble Voting