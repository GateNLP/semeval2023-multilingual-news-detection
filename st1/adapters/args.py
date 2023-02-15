from typing import Optional

import dataclasses
from dataclasses import dataclass, field
import wandb


class BaseArguments:
    def wandb_push(self):
        wandb.config.update(dataclasses.asdict(self))


@dataclass
class DataArguments(BaseArguments):
    merge_and_resplit: Optional[bool] = field(default=True)
    external_satire: Optional[bool] = field(default=False)
    external_satire_translated: Optional[bool] = field(default=False)
    hold_out_lang: Optional[str] = field(default=None)
    stratify_field: Optional[str] = field(default='labels')


@dataclass
class ModelArguments(BaseArguments):
    name: Optional[str] = field(default="bert-base-multilingual-cased")
    adapter_dropping: Optional[bool] = field(default=False)
    adapter_reduction_factor: Optional[int] = field(default=3)
    adapter_config_type: Optional[str] = field(default='base',
                                               metadata={"help": "'base' or 'pfeiffer'"})


@dataclass
class MyTrainingArguments(BaseArguments):
    learning_rate: Optional[float] = field(default=1e-4)
    warmup_ratio: Optional[float] = field(default=0)
    weight_decay: Optional[float] = field(default=0)
    num_train_epochs: Optional[float] = field(default=20)
    load_best_model_at_end: Optional[bool] = field(default=True)