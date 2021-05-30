import yaml
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

from dataset import all_dataset
from model import QA_Model, Risk_Model
from utils.init import set_random_seed, get_device

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)

set_random_seed(configs["seed"])
torch_device = get_device()


def train(model, train_loader, val_loader=None):
    model.train()
    model.to(torch_device)
    optimizer = AdamW(model.parameters(), lr=configs["learning_rate"])

    avg_loss = 0
    for epoch in range(configs["epochs"]):
        tqdm_train_loader = tqdm(train_loader)
        for step, batch in enumerate(tqdm_train_loader):
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            answer = batch["answer"].float().to(torch_device)

            optimizer.zero_grad()
            loss = model.loss_func(input_ids, attention_mask, answer)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if step % configs["log_step"] == 0:
                avg_loss = avg_loss / configs["log_step"]
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch:03}] Loss:{avg_loss:.3f}"
                )
                # writer.add_scalar("loss", avg_loss, step)
                avg_loss = 0

        # if(val_loader is not None):
        #     evaluate(model, val_loader)
        #     tqdm_train_loader.set_description(
        #         f"[Epoch:{epoch:03}] Loss:{avg_loss:.3f}"
        #     )

    return model


def evaluate(model, val_loader):
    model.eval()
    model.to(torch_device)
    loss = 0
    for batch in val_loader:
        batch_document = [raw_data.article[idx] for idx in batch["article_id"]]
        batch_document = torch.LongTensor(batch_document).to(torch_device)
        batch_question = batch["question"].to(torch_device)
        batch_choice = batch["choice"].to(torch_device)
        batch_qa_answer = batch["qa_answer"].float().to(torch_device)
    return loss


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

    qa_model = train(QA_Model(), train_loader, val_loader)
