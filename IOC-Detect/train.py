import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.datasets import CTIDatasetForBiLSTMCRF, Data2Idx


def main() -> None:
    # device = torch.device('mps' if torch.has_mps else 'cpu')        # for mac
    # device = torch.device('cuda' if torch.cuda.device else 'cpu')   # for windows
    device = torch.device('cpu')                                    # a speical case for my mac
    
    # Parameters
    train_path = '../data/cti_train_data.csv'
    test_path = '../data/cti_test_data.csv'
    ckp_path = './log'
    model_path = './model_save'
    
    # Check path
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Hyperparameters
    N_GRAM = 3
    SEQ_LEN = 100
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 128
    N_LAYERS = 2
    BATCH_SIZE = 32
    N_EPOCHS = 10
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    
    # Load data
    train_data = pd.read_csv(train_path, encoding='utf-8').dropna()
    test_data = pd.read_csv(test_path, encoding='utf-8').dropna()
    
    # Get dataset
    train_dataset = CTIDatasetForBiLSTMCRF(sentences=train_data['Sentences'].values,
                                           tags=train_data['Tags'].values,
                                           sequence_len=SEQ_LEN,
                                           transforms=Data2Idx,
                                           target_transforms=Data2Idx)
    test_dataset = CTIDatasetForBiLSTMCRF(sentences=test_data['Sentences'].values,
                                          tags=test_data['Tags'].values,
                                          sequence_len=SEQ_LEN,
                                          transforms=Data2Idx,
                                          target_transforms=Data2Idx)
    
    # Get dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             drop_last=True)


if __name__ == '__main__':
    main()
