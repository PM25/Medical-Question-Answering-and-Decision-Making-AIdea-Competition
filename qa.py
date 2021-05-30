import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from transformers import get_linear_schedule_with_warmup

from dataset import all_dataset
from model import QA_Model, Risk_Model
from utils.init import set_random_seed, get_device

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)

set_random_seed(configs["seed"])
torch_device = get_device(0)
torch.cuda.empty_cache()


def train(model, train_loader, val_loader=None):
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

    Path(configs["log_path"]).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=configs["log_path"])

    for epoch in range(configs["epochs"]):
        tqdm_train_loader = tqdm(train_loader)
        avg_loss = 0
        for step, batch in enumerate(tqdm_train_loader, 1):
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            answer = batch["answer"].float().to(torch_device)

            optimizer.zero_grad()
            pred, loss = model.pred_and_loss(input_ids, attention_mask, answer)
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()

            if val_loader is not None and step == len(train_loader):
                val_loss, val_acc = evaluate(model, val_loader)
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch:03}] Train Loss: {avg_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}",
                )

            elif step % configs["log_step"] == 0:
                avg_loss /= configs["log_step"]
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch}] Train Loss:{avg_loss:.3f}"
                )
                writer.add_scalar("loss", avg_loss, step)
                avg_loss = 0

    return model


def evaluate(model, val_loader):
    model.eval()
    model.to(torch_device)

    val_acc, val_loss = [], []
    for step, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        answer = batch["answer"].float().to(torch_device)

        preds, loss = model.pred_and_loss(input_ids, attention_mask, answer)
        labels = torch.argmax(answer, dim=1)
        val_acc.append((preds == labels).cpu().numpy().mean())
        val_loss.append(loss.item())

    val_acc = np.mean(val_acc)
    val_loss = np.mean(val_loss)

    return val_loss, val_acc


if __name__ == "__main__":
    dataset = all_dataset(configs["qa_data"], configs["risk_data"])

    val_size = int(len(dataset) * configs["val_size"])
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], num_workers=4
    )

    qa_model = train(QA_Model(freeze_bert=True), train_loader, val_loader)
