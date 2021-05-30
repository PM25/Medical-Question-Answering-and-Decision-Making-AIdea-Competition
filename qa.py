import yaml
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import random_split

from dataset import all_dataset
from model import QA_Model, Risk_Model
from utils.init import set_random_seed, get_device

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)

set_random_seed(configs["seed"])
torch_device = get_device()


def train(model, raw_data, train_loader, val_loader=None):
    model.train()
    model.to(torch_device)
    optimizer = AdamW(model.parameters(), lr=configs["learning_rate"])

    for epoch in range(configs["epochs"]):
        tqdm_train_loader = tqdm(train_loader)
        for batch in train_loader:
            batch_document = [raw_data.article[idx] for idx in batch["article_id"]]
            batch_document = torch.LongTensor(batch_document).to(torch_device)
            batch_question = batch["question"].to(torch_device)
            batch_choice = batch["choice"].to(torch_device)
            batch_qa_answer = batch["qa_answer"].float().to(torch_device)

            optimizer.zero_grad()
            loss = model.loss_fn(
                batch_document, batch_question, batch_choice, batch_qa_answer
            )
            loss.backward()
            optimizer.step()

            if epoch % configs["log_steps"] == 0:
                avg_loss = avg_loss / configs["log_step"]
                tqdm_train_loader.set_description(f"epoch:{epoch}, loss:{avg_loss}")
                # writer.add_scalar("loss", avg_loss, step)
                avg_loss = 0


def evaluate(model, raw_data, val_loader):
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
    
    # val_dataset, train_dataset = random_split(
    #     dataset, [configs["val_size"], 1 - configs["val_size"]]
    # )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=configs["batch_size"], shuffle=True
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=configs["batch_size"], shuffle=True
    # )

    # qa_model = QA_Model()
    # train(qa_model, dataset, train_loader, val_loader)
