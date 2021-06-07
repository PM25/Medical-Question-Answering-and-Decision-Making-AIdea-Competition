import csv
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from transformers import get_linear_schedule_with_warmup, logging

from dataset import risk_dataset
from model.risk_model import Classifier
from utils.setting import set_random_seed, get_device


def train(args, configs, model, train_loader, val_loader):
    model.train()
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=configs["learning_rate"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=configs["warmup_steps"],
        num_training_steps=len(train_loader) * configs["epochs"],
    )

    writer = SummaryWriter(log_dir=args.expdir)

    best_val_acc = 0
    best_pth = None
    for epoch in range(configs["epochs"]):
        avg_loss, total_loss = 0, 0
        tqdm_train_loader = tqdm(train_loader)
        for step, batch in enumerate(tqdm_train_loader, 1):
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(args.device)

            optimizer.zero_grad()
            preds = model(**batch)
            loss = F.binary_cross_entropy(preds, batch["labels"].float())

            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()
            total_loss += loss.item()

            if val_loader is not None and step == len(train_loader):
                val_loss, val_acc = evaluate(args, configs, model, val_loader)
                train_loss = total_loss / len(train_loader)
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch:03}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}",
                )
                writer.add_scalar("Risk_Accuracy/valalidation", val_acc, epoch)
                writer.add_scalar("Risk_Loss/validation", val_loss, epoch)
                writer.add_scalar("Risk_Loss/train", train_loss, epoch)

                if val_acc > best_val_acc:
                    best_pth = f"{args.expdir}/{val_acc}-{val_loss}.pth"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "args": args,
                            "config": configs,
                        },
                        best_pth,
                    )
                    best_val_acc = val_acc

            elif step % configs["log_step"] == 0:
                avg_loss /= configs["log_step"]
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch}] Train Loss:{avg_loss:.3f}"
                )
                avg_loss = 0

    writer.close()
    assert best_pth is not None
    return best_pth


def evaluate(args, configs, model, val_loader):
    model.eval()
    model.to(args.device)

    val_loss = []
    all_preds, truth = [], []
    for step, batch in enumerate(val_loader):
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(args.device)

        preds = model(**batch)
        loss = F.binary_cross_entropy(preds, batch["labels"].float())

        all_preds.extend(preds.cpu().tolist())
        truth.extend(batch["labels"].cpu().tolist())
        val_loss.append(loss.item())

    val_loss = np.mean(val_loss)
    score = roc_auc_score(truth, all_preds) if len(truth) != 0 else np.nan

    return val_loss, score


def save_preds(args, configs, model, data_loader):
    model.eval()
    model.to(args.device)

    all_preds = []
    for step, batch in enumerate(data_loader):
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(args.device)

        preds = model(**batch)
        all_preds.extend(list(zip(batch["ids"].cpu().tolist(), preds.cpu().tolist())))

    with open(f"{args.expdir}/decision.csv", "w") as f:
        csvwriter = csv.writer(f, delimiter=",")
        csvwriter.writerow(["article_id", "probability"])
        for article_id, prob in all_preds:
            csvwriter.writerow([article_id, prob])

    print("*Successfully saved prediction to output/risk.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--config", default="configs.yaml")
    parser.add_argument("--expdir")
    parser.add_argument("--test_pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=1009)
    args = parser.parse_args()

    set_random_seed(args.seed)
    torch.cuda.empty_cache()
    logging.set_verbosity(logging.ERROR)

    if args.mode == "train":
        assert args.expdir is not None
        Path(args.expdir).mkdir(parents=True, exist_ok=True)

        with open(args.config, "r") as stream:
            configs = yaml.safe_load(stream)

        with open(f"{args.expdir}/configs.yaml", "w") as yaml_file:
            yaml.dump(configs, yaml_file, default_flow_style=False)

        dataset = risk_dataset(configs["risk_data"], **configs)
        val_size = int(len(dataset) * configs["val_size"])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=configs["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=risk_dataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=risk_dataset.collate_fn,
        )

        risk_model = Classifier(**configs)
        best_pth = train(args, configs, risk_model, train_loader, val_loader)

        test_dataset = risk_dataset(configs["dev_risk_data"], **configs)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=risk_dataset.collate_fn,
        )

        best_states = torch.load(best_pth)
        risk_model.load_state_dict(best_states["model"])
        save_preds(args, configs, risk_model, test_loader)

    elif args.mode == "test":
        best_states = torch.load(args.test_pth)
        args, configs = best_states["args"], best_states["configs"]

        test_dataset = risk_dataset(configs["dev_risk_data"], **configs)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=risk_dataset.collate_fn,
        )

        risk_model = Classifier(**configs)
        risk_model.load_state_dict(best_states["model"])
        save_preds(args, configs, risk_model, test_loader)


if __name__ == "__main__":
    main()
