import csv
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from transformers import get_linear_schedule_with_warmup, logging

from dataset import qa_dataset
from model import get_qa_model
from utils.setting import set_random_seed, get_device

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)

set_random_seed(configs["seed"])
torch_device = get_device(configs["device_id"])
torch.cuda.empty_cache()
logging.set_verbosity(logging.ERROR)


def train(model, train_loader, val_loader=None, configs=configs):
    model.train()
    model.to(torch_device)

    optimizer = AdamW(
        model.parameters(),
        lr=configs["learning_rate"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=configs["warmup_steps"],
        num_training_steps=len(train_loader) * configs["epochs"],
    )

    writer = SummaryWriter()

    for epoch in range(configs["epochs"]):
        avg_loss, train_loss = 0, 0
        tqdm_train_loader = tqdm(train_loader)
        for step, batch in enumerate(tqdm_train_loader, 1):
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(torch_device)

            optimizer.zero_grad()
            preds, loss = model(**batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()
            train_loss += loss.item()

            if val_loader is not None and step == len(train_loader):
                val_loss, val_acc = evaluate(model, val_loader)
                train_loss /= len(train_loader)
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch:03}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}",
                )
                writer.add_scalar("QA_Accuracy/valalidation", val_acc, epoch)
                writer.add_scalar("QA_Loss/validation", val_loss, epoch)
                writer.add_scalar("QA_Loss/train", train_loss, epoch)

            elif step % configs["log_step"] == 0:
                avg_loss /= configs["log_step"]
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch}] Train Loss:{avg_loss:.3f}"
                )
                avg_loss = 0

    writer.close()
    return model


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    model.to(torch_device)

    val_acc, val_loss = [], []
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(torch_device)

            preds, loss = model(**batch)
            preds = torch.argmax(preds, dim=-1)
            val_acc.append((preds == batch["answer"]).cpu().numpy().mean())
            val_loss.append(loss.item())

    val_acc = np.mean(val_acc)
    val_loss = np.mean(val_loss)
    model.train()

    model.train()
    return val_loss, val_acc


@torch.no_grad()
def save_preds(model, data_loader):
    model.eval()
    model.to(torch_device)

    all_preds = []
    for step, batch in enumerate(data_loader):
        _id = batch["id"]
        labels = batch["label"]
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)

        preds = model.pred_label(input_ids, attention_mask, labels)
        all_preds.extend(list(zip(_id.tolist(), preds)))

    Path("output").mkdir(parents=True, exist_ok=True)
    with open("output/qa.csv", "w") as f:
        csvwriter = csv.writer(f, delimiter=",")
        csvwriter.writerow(["id", "answer"])
        for _id, pred in all_preds:
            csvwriter.writerow([_id, pred])
    with open("output/qa_configs.yml", "w") as yaml_file:
        yaml.dump(configs, yaml_file, default_flow_style=False)
    print("*Successfully saved prediction to output/qa.csv")


if __name__ == "__main__":
    dataset = qa_dataset(configs, configs["qa_data"])

    val_size = int(len(dataset) * configs["val_size"])
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=0,
        collate_fn=qa_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], num_workers=4,
        collate_fn=qa_dataset.collate_fn,
    )

    qa_model = train(get_qa_model(**configs), train_loader, val_loader)

    test_dataset = qa_dataset(configs, configs["dev_qa_data"])
    test_loader = DataLoader(
        test_dataset, batch_size=configs["batch_size"], num_workers=4,
        collate_fn=qa_dataset.collate_fn,
    )
    save_preds(qa_model, test_loader)
