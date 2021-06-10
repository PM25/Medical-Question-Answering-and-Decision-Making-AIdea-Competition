import csv
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from transformers import get_linear_schedule_with_warmup, logging

from dataset import risk_dataset
from model import BertClassifier
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
        avg_loss, total_loss = 0, 0
        tqdm_train_loader = tqdm(train_loader)
        for step, batch in enumerate(tqdm_train_loader, 1):
            article = [b['article'] for b in batch]
            answer = torch.Tensor([b['label'] for b in batch]).float().to(torch_device)
            optimizer.zero_grad()
            preds, loss = model(article, torch_device, answer)
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()
            total_loss += loss.item()

            if val_loader is not None and step == len(train_loader):
                val_loss, val_acc = evaluate(model, val_loader)
                train_loss = total_loss / len(train_loader)
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch:03}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}",
                )
                writer.add_scalar("Risk_Accuracy/valalidation", val_acc, epoch)
                writer.add_scalar("Risk_Loss/validation", val_loss, epoch)
                writer.add_scalar("Risk_Loss/train", train_loss, epoch)

            elif step % configs["log_step"] == 0:
                avg_loss /= configs["log_step"]
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch}] Train Loss:{avg_loss:.3f}"
                )
                avg_loss = 0
        
        if epoch + 1 >= configs["warmup_epoch"]:
            for param in model.pretrained_model.bert.parameters():
                param.requires_grad = True
            
    writer.close()
    return model


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        val_loss = []
        all_preds, truth = [], []
        for step, batch in enumerate(val_loader):
            article = [b['article'] for b in batch]
            answer = torch.Tensor([b['label'] for b in batch]).float().to(torch_device)
            preds, loss = model(article, torch_device, answer)
            all_preds.extend(preds.tolist())
            truth.extend(answer.tolist())
            val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        score = roc_auc_score(truth, all_preds) if len(truth) != 0 else np.nan
    model.train()
    return val_loss, score


@torch.no_grad()
def save_preds(model, data_loader):
    model.eval()
    model.to(torch_device)

    all_preds = []
    for step, batch in enumerate(data_loader):
        article_ids = [b['article_id'] for b in batch]
        article = [b['article'] for b in batch]
        preds = model(article, torch_device)
        all_preds.extend(list(zip(article_ids, preds.tolist())))

    Path("output").mkdir(parents=True, exist_ok=True)
    with open("output/decision.csv", "w") as f:
        csvwriter = csv.writer(f, delimiter=",")
        csvwriter.writerow(["article_id", "probability"])
        for article_id, prob in all_preds:
            csvwriter.writerow([article_id, prob])
    with open("output/decision_configs.yml", "w") as yaml_file:
        yaml.dump(configs, yaml_file, default_flow_style=False)
    print("Successfully saved prediction to output/decision.csv")


if __name__ == "__main__":
    dataset = risk_dataset(configs, configs["risk_data"])

    val_size = int(len(dataset) * configs["val_size"])
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.test = False
    
    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=1, collate_fn=dataset.collate_fn
    )

    risk_model = train(BertClassifier(configs), train_loader, val_loader)
    
    test_dataset = risk_dataset(configs, configs["dev_risk_data"])
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=1, collate_fn=test_dataset.collate_fn
    )
    save_preds(risk_model, test_loader)