# SheffieldVeraAi at SemEval-2023 Task 3

This is the source code for _SheffieldVeraAI at SemEval-2023 Task 3: Mono and Multilingual Approaches for News Genre, Topic and Persuasion Technique Classification_ ([task](https://propaganda.math.unipd.it/semeval2023task3/), [paper](https://aclanthology.org/2023.semeval-1.275.pdf)).

Authors are members of the [GATE](https://gate.ac.uk) team of the [University of Sheffield Natural Language Processing group](https://www.sheffield.ac.uk/dcs/research/groups/nlp). 

Our models performed well across the board, achieving the highest performance in some languages, including zero-shot languages, and the highest mean across languages for sub-tasks 1 and 2.

## Setup
Download the organiser data and extract the data dir into `./data/`, i.e. each language's data should be at `./data/{LANG}/`.

The code to train each subtask is split in the `st1`, `st2` and `st3` directories.

## Acknowledgements
This work has been co-funded by the European Union under the Horizon Europe vera.ai (grant 101070093) and Vigilant (grant 101073921) projects and the UK’s innovation agency (InnovateUK) grants 10039055 and 10039039.
