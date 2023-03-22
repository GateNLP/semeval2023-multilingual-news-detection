"""Train and log transformer models using wandb."""

import torch
import sys
import argparse
from datetime import datetime
from bert_utils import SemEvalDataModule, SemEvalTransformer

from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--train_languages",
        type=str,
        required=True,
        nargs="*",
        choices=["en", "fr", "ge", "it", "po", "ru"],
    )
    parser.add_argument(
        "--val_languages",
        type=str,
        required=False,
        nargs="*",
        choices=["en", "fr", "ge", "it", "po", "ru"],
    )
    parser.add_argument("--pretrained_name", type=str, required=True)
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--classification_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="loss",
        choices=["loss", "f1_micro", "f1_macro"],
    )
    parser.add_argument("--with_spans", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--with_unlabelled", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--with_translations", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--with_class_weights", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--upsample", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--offline", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--log_model", type=bool, action=argparse.BooleanOptionalAction)

    return parser.parse_args(argv)


if __name__ == "__main__":
    now_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args = parse_args(sys.argv[1:])
    seed_everything(args.seed, workers=True)

    dm = SemEvalDataModule(
        train_languages=args.train_languages,
        val_languages=args.val_languages,
        pretrained_name=args.pretrained_name,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        input_col="text",
        output_col="labels",
        with_spans=args.with_spans,
        with_translations=args.with_translations,
        upsample=args.upsample,
        with_unlabelled=args.with_unlabelled,
    )

    dm.prepare_data()

    model = SemEvalTransformer(
        pretrained_name=args.pretrained_name,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        class_weights=dm.class_weights if args.with_class_weights else None,
        classification_threshold=args.classification_threshold,
    )

    timer_callback = Timer()
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor_metric,
        save_top_k=1,
        filename=args.experiment_name + "_" + "{epoch}-{loss:.2f}" + "%" + now_timestamp,
        mode="min" if args.monitor_metric == "loss" else "max",
    )
    logger = WandbLogger(
        project=args.project_name,
        name=args.experiment_name,
        log_model=not args.offline and args.log_model,
        tags=[args.train_languages[0] if len(args.train_languages) == 1 else "multi"],
        offline=args.offline,
    )
    logger.experiment.config.update(
        {
            "seed": args.seed,
            "train_languages": args.train_languages,
            "val_languages": args.val_languages,
            "upsample": args.upsample,
            "with_translations": args.with_translations,
        }
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        # gradient_clip_algorithm="norm",
        # gradient_clip_val=1.0,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, timer_callback],
        logger=logger,
        limit_val_batches=0 if args.val_languages is None else 1.0,
        num_sanity_val_steps=0 if args.val_languages is None else 2,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)

    print(f"Train Time: {timer_callback.time_elapsed('train'):.2f}s")
    print(f"Validation Time: {timer_callback.time_elapsed('validate'):.2f}s")
