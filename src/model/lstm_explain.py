import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import (classification_report, f1_score,
#                              precision_score, recall_score,
#                              average_precision_score,
#                              roc_auc_score)
import json
# import captum

SEED = 4266
DATA_DIR = "../../data/processed"
# DROP_OUT = 0.3
# LEARNING_RATE = 1e-3
# HIDDEN_DIM = 100
INPUT_DIM = 59
OUTPUT_DIM = 5
# LAYER_DIM = 1
EPOCHS = 20
PATIENCE = 4
MIN_DELTA = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim  = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
                            batch_first=True,
                            dropout=dropout if layer_dim > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.attn    = nn.Linear(hidden_dim, 1)
        self.fc      = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #we reset the hidden state for each batch of sequences so that they are independent of each other. This is important for training on batches of sequences.
        out, _ = self.lstm(x)                       # (B, T, H) — zeros state by default
        # attention pooling over time
        scores  = self.attn(out).squeeze(-1)         # (B, T)
        #weight of each timestep
        weights = torch.softmax(scores, dim=1)       # (B, T)
        #weighted average
        pooled  = (out * weights.unsqueeze(-1)).sum(1)  # (B, H)
        #regularisation
        pooled  = self.dropout(pooled)
        return self.fc(pooled)

def load_baseline():
    hidden_dim = 128
    layer_dim = 2
    dropout = 0.2
    lr = 1e-3
    weight_decay = 0
    epochs = 100
    seed = SEED
    patience = 10
    optimizer_name = "adam"
    batch_size = 128
    min_delta = 0

    #set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)


    #create LSTM
    model = LSTMModel(INPUT_DIM, hidden_dim, layer_dim, OUTPUT_DIM, dropout).to(device)
    model.load_state_dict(torch.load("lstm_model.pt", map_location=device))##, weights_only=True))
    #disable dropout
    model.eval()
    return model


baseline_model = load_baseline()

