import yaml
import pprint
import datetime
from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader

from model import BertClassifier
from dataset import risk_dataset
from risk import train, evaluate, save_preds
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

        risk_model = train(model(configs), train_loader, val_loader, configs=configs)
        val_loss, val_acc = evaluate(risk_model, val_loader)

        Path("history").mkdir(parents=True, exist_ok=True)
        _id = datetime.datetime.now().strftime("%d%H%M%S")
        with open(f"history/risk_val_acc_{val_acc:.3f}_id_{_id}.yml", "w") as yaml_file:
            yaml.dump(configs, yaml_file, default_flow_style=False)

        return val_acc, configs

    for hypeparam, _range in list(grid_config.items()):
        grid_config.pop(hypeparam, None)
        for value in _range:
            configs[hypeparam] = value
            acc, configs = grid_search(
                configs, grid_config, model, train_loader, val_loader
            )
            if acc < best_acc:
                best_acc = acc
                best_configs = configs
        break

    return best_acc, best_configs


def get_dataloader(configs, train_file, test_file):
    dataset = risk_dataset(configs, train_file)
    test_dataset = risk_dataset(configs, test_file)

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
        configs, configs["risk_data"], configs["dev_risk_data"]
    )

    grid_seach_config = {
        "epochs": [1, 5, 10, 15, 20, 25, 30, 35],
        "batch_size": [1],
        "freeze_bert": [True],
        "learning_rate": [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "warmup_steps": [0, 50, 100, 150],
        "n_cls_layers": [1, 2, 3, 4, 5],
        "hidden_dim": [50, 100, 150, 200, 300, 400, 500],
    }

    best_acc, best_configs = grid_search(
        configs, grid_seach_config, BertClassifier, train_loader, val_loader
    )
    best_configs["val_size"] = 0
    final_risk_model = train(
        BertClassifier(best_configs), all_loader, configs=best_configs
    )

    save_preds(final_risk_model, test_loader)