from bert_utils import SemEvalTransformer, SemEvalDataModule
import sys
import argparse
import os
from pytorch_lightning.loggers import WandbLogger
from utils import load_dataset, multibin
import torch
import numpy as np
import pandas as pd
from datetime import datetime


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument(
        "--test_language",
        type=str,
        required=True,
        choices=["en", "fr", "ge", "it", "po", "ru", "gr", "ka", "es"],
    )
    parser.add_argument("--model_version", type=str, required=False, default="best_k")
    parser.add_argument("--classification_threshold", type=float, default=0.5)

    return parser.parse_args(argv)


if __name__ == "__main__":
    now_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args = parse_args(sys.argv[1:])
    checkpoint_reference = (
        f"jaugusto97/{args.project_name}/model-{args.run_id}:{args.model_version}"
    )
    artifact_dir = WandbLogger.download_artifact(artifact=checkpoint_reference)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = (
        SemEvalTransformer.load_from_checkpoint(
            os.path.join(artifact_dir, "model.ckpt")
        )
        .to(device)
        .eval()
    )
    test_df = load_dataset([args.test_language], split="test").set_index(["id", "line"])

    dm = SemEvalDataModule(
        pretrained_name=model.hparams.pretrained_name,
        test_language=args.test_language,
        batch_size=model.hparams.batch_size,
        max_seq_len=model.hparams.max_seq_len,
        input_col="text",
        output_col="labels",
    )
    dm.prepare_data()

    full_pred = []
    for batch in dm.test_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs["logits"]

        probs = torch.nn.Sigmoid()(logits)
        preds = torch.zeros(probs.shape)
        preds[torch.where(probs >= args.classification_threshold)] = 1

        full_pred.append(preds.detach().cpu().numpy())

    preds = np.concatenate(full_pred)
    out = multibin.inverse_transform(preds)
    out = list(map(lambda x: ",".join(x), out))
    out = pd.DataFrame(out, test_df.index)
    if not os.path.exists(f"submissions/{args.test_language}"):
        os.makedirs(f"submissions/{args.test_language}")

    out.to_csv(
        f"submissions/{args.test_language}/final_submission_{now_timestamp}_{args.project_name}_{args.run_id}_{args.model_version}.txt",
        sep="\t",
        header=None,
    )
