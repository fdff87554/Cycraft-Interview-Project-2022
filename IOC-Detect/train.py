import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model.bilstm_crf import BiLSTM_CRF
from utils.datasets import CTIDatasetForBiLSTMCRF, Data2Idx


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int, learning_rate: float,
          optimizer: str, device: torch.device, save_path: str, save_name: str) -> None:
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Optimizer must be Adam or SGD')

    for epoch in range(1, epochs+1):
        model.train().to(device)
        train_loss = 0
        for sentences, tags, mask in tqdm(train_loader):
            sentences = sentences.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            loss = model(sentences, tags, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(train_loader)}')

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval().to(device)
                test_loss = 0
                for sentences, tags, mask in tqdm(test_loader):
                    sentences = sentences.to(device)
                    tags = tags.to(device)
                    mask = mask.to(device)

                    loss = model(sentences, tags, mask)
                    test_loss += loss.item()
            print(f'Epoch: {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Test Loss: {test_loss / len(test_loader)}')
            torch.save(model.state_dict(), os.path.join(save_path, save_name))


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
    
    # Get model
    model = BiLSTM_CRF(vocab_size=train_dataset.vocab_size,
                       tag_size=train_dataset.tag_size,
                       embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM,
                       n_layers=N_LAYERS,
                       batch_size=BATCH_SIZE,
                       dropout=DROPOUT).to(device)
    
    # Train
    train(model=model,
          train_loader=train_loader,
          test_loader=test_loader,
          epochs=N_EPOCHS,
          learning_rate=LEARNING_RATE,
          optimizer='Adam',
          device=device,
          save_path=ckp_path,
          save_name=f'bilstm_crf_{time.time()}.pt')
    
    # Save model
    torch.save(model.state_dict(), os.path.join(model_path, 'bilstm_crf.pt'))
    
    # Test
    with torch.no_grad():
        model.eval()
        for sentences, tags, mask in tqdm(test_loader):
            sentences = sentences.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            pred = model(sentences, mask=mask)
            tags = tags.cpu().numpy()
            
            for i in range(len(pred)):
                ans = ''
                preds = ''
                p_tag = ''
                for j in range(len(pred[i])):
                    if mask[i][j] == 1:
                        ans = ans + ' ' + train_dataset.idx2vocab[int(sentences[i][j])]
                        preds = preds + ' ' + train_dataset.idx2tag[pred[i][j]]
                        p_tag = p_tag + ' ' + train_dataset.idx2tag[tags[i][j]]
                print(f'Word: {ans}')
                print(f'Pred: {preds}')
                print(f'True: {p_tag}')


if __name__ == '__main__':
    main()
