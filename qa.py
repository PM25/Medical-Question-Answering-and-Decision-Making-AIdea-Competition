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

from dataset import *
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
        avg_loss, train_loss, train_acc = 0, 0, []

        tqdm_train_loader = tqdm(train_loader)
        for step, batch in enumerate(tqdm_train_loader, 1):
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(torch_device)

            optimizer.zero_grad()
            logits, acc, loss = model(**batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()
            train_loss += loss.item()
            train_acc += acc

            if val_loader is not None and step == len(train_loader):
                val_loss, val_acc = evaluate(model, val_loader)
                train_loss /= len(train_loader)
                train_acc = np.mean(train_acc)
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch:03}] Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}",
                )
                writer.add_scalar("QA_Accuracy/valalidation", val_acc, epoch)
                writer.add_scalar("QA_Loss/validation", val_loss, epoch)
                writer.add_scalar("QA_Accuracy/train", train_acc, epoch)
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
    qa_ids, logits, is_answers = [], [], []
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(torch_device)

            logit, acc, loss = model(**batch)
            val_acc += acc
            val_loss.append(loss.item())

            logits += logit
            qa_ids += batch["qa_id"]
            is_answers += batch["is_answer"].cpu().tolist()

    if isinstance(val_loader.dataset, qa_binary_dataset):
        final_predict = defaultdict(list)
        final_answer = dict()
        for qa_id, logit, is_answer in zip(qa_ids, logits, is_answers):
            final_predict[qa_id].append(logit)
            if is_answer:
                final_answer[qa_id] = len(final_predict[qa_id]) - 1

        final_acc = []
        for qa_id in final_predict.keys():
            if final_answer[qa_id] == torch.tensor(final_predict[qa_id]).argmax().item():
                final_acc.append(1)
            else:
                final_acc.append(0)
        val_acc = final_acc

    val_acc = np.mean(val_acc)
    val_loss = np.mean(val_loss)
    model.train()

    return val_loss, val_acc


@torch.no_grad()
def save_preds(model, data_loader):
    model.eval()
    model.to(torch_device)

    all_preds = []
    qa_ids, logits, labels = [], [], []
    for step, batch in enumerate(data_loader):
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(torch_device)

        logit = model.infer(**batch)
        logits += logit
        qa_ids += batch["qa_id"]
        labels += batch["label"]
        
        if isinstance(data_loader.dataset, qa_multiple_dataset):
            start_indices = list(range(0, len(batch["qa_id"]), 3))
            pred_idx = torch.tensor(logit).argmax(dim=-1).view(-1)
            assert len(start_indices) == len(pred_idx)
            for idx, start in zip(pred_idx, start_indices):
                candidate = batch["label"][start : start + 3]
                qa_id = batch["qa_id"][start : start + 3]
                all_preds.append((qa_id[0], candidate[idx]))

    if isinstance(data_loader.dataset, qa_binary_dataset):
        final_predict = defaultdict(list)
        for qa_id, logit, label in zip(qa_ids, logits, labels):
            final_predict[qa_id].append((logit, label))

        final_answer = dict()
        for qa_id in final_predict.keys():
            assert len(final_predict[qa_id]) == 3
            logits, labels = zip(*final_predict[qa_id])
            best_id = torch.tensor(logits).argmax().item()
            answer = labels[best_id]
            final_answer[qa_id] = answer

        all_preds = []
        for i in sorted(final_answer.keys()):
            all_preds.append((i, final_answer[i]))

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
    qa_dataset = eval(configs["dataset_class"])
    train_dataset = qa_dataset(configs, configs["qa_data"], training=True)
    val_dataset = qa_dataset(configs, configs["qa_data"], training=False)

    # val_size = int(len(dataset) * configs["val_size"])
    # train_size = len(dataset) - val_size

    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=0,
        collate_fn=qa_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], num_workers=4,
        collate_fn=qa_dataset.collate_fn,
    )

    qa_model = train(get_qa_model(configs), train_loader, val_loader)
    # qa_model = get_qa_model(configs)

    test_dataset = qa_dataset(configs, configs["dev_qa_data"])
    test_loader = DataLoader(
        test_dataset, batch_size=configs["batch_size"], num_workers=4,
        collate_fn=qa_dataset.collate_fn,
    )
    save_preds(qa_model, test_loader)
