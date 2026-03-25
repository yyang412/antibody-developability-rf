import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import spearmanr


def spearman_corr(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy().reshape(-1)
    y_pred = y_pred.detach().cpu().numpy().reshape(-1)
    return spearmanr(y_true, y_pred)[0]


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    preds = []
    trues = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = F.mse_loss(out.squeeze(), batch.y.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds.append(out.detach().cpu())
        trues.append(batch.y.detach().cpu())

    preds = torch.cat(preds).squeeze()
    trues = torch.cat(trues).squeeze()

    avg_loss = total_loss / len(loader.dataset)
    rho = spearman_corr(trues, preds)

    return avg_loss, rho


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    preds = []
    trues = []

    for batch in loader:
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = F.mse_loss(out.squeeze(), batch.y.squeeze())

        total_loss += loss.item() * batch.num_graphs
        preds.append(out.detach().cpu())
        trues.append(batch.y.detach().cpu())

    preds = torch.cat(preds).squeeze()
    trues = torch.cat(trues).squeeze()

    avg_loss = total_loss / len(loader.dataset)
    rho = spearman_corr(trues, preds)

    return avg_loss, rho