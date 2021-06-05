import yaml
import pprint
import datetime
from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader

from model import QA_Model
from dataset import qa_dataset
from qa import train, evaluate, save_preds
from utils.setting import set_random_seed, get_device

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)

set_random_seed(configs["seed"])
torch_device = get_device(configs["device_id"])
torch.cuda.empty_cache()

pp = pprint.PrettyPrinter(depth=4)


def grid_search(configs, grid_config, model, train_loader, val_loader):
    best_acc, best_configs = 0, configs

    if len(grid_config) == 0:
        pp.pprint(configs)

        qa_model = train(model(configs), train_loader, val_loader, configs=configs)
        val_loss, val_acc = evaluate(qa_model, val_loader)

        Path("history").mkdir(parents=True, exist_ok=True)
        _id = datetime.datetime.now().strftime("%d%H%M%S")
        with open(f"history/qa_val_acc_{val_acc:.3f}_id_{_id}.yaml", "w") as yaml_file:
            yaml.dump(configs, yaml_file, default_flow_style=False)

        return val_acc, configs

    for hypeparam, _range in list(grid_config.items()):
        grid_config.pop(hypeparam, None)
        for value in _range:
            configs[hypeparam] = value
            acc, configs = grid_search(
                configs.copy(), grid_config.copy(), model, train_loader, val_loader
            )
            if acc < best_acc:
                best_acc = acc
                best_configs = configs
        break

    return best_acc, best_configs


def get_dataloader(configs, train_file, test_file):
    dataset = qa_dataset(configs, train_file)
    test_dataset = qa_dataset(configs, test_file)

    val_size = int(len(dataset) * configs["val_size"])
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    all_loader = DataLoader(
        dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=8
    )
    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=configs["batch_size"], num_workers=8
    )

    return all_loader, train_loader, val_loader, test_loader


if __name__ == "__main__":
    all_loader, train_loader, val_loader, test_loader = get_dataloader(
        configs, configs["qa_data"], configs["dev_qa_data"]
    )

    grid_seach_config = {
        "model": ["Roberta"],
        "epochs": [1, 2, 3, 4],
        "batch_size": [1],
        "freeze_bert": [False],
        "learning_rate": [1e-3, 1e-4, 1e-5],
        "warmup_steps": [0, 50, 150, 300],
        "n_cls_layers": [1, 3, 5, 7],
        "hidden_dim": [50, 100, 200, 400, 700],
    }

    best_acc, best_configs = grid_search(
        configs, grid_seach_config, QA_Model, train_loader, val_loader
    )
    best_configs["val_size"] = 0
    final_qa_model = train(QA_Model(best_configs), all_loader, configs=best_configs)

    save_preds(final_qa_model, test_loader)