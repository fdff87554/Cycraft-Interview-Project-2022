import os

import numpy as np
import pandas as pd

import torch

from utils.datasets import DependancyParserDataset


# def detect_relations(sentence: str, tag: str) -> list[tuple[str, str]]:


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
    
    
    # Load data
    train_data = pd.read_csv(train_path, encoding='utf-8').dropna()
    test_data = pd.read_csv(test_path, encoding='utf-8').dropna()
    
    # Get dataset
    train_dataset = DependancyParserDataset(sentences=train_data['Sentences'].values,
                                            tags=train_data['Tags'].values)
    test_dataset = DependancyParserDataset(sentences=test_data['Sentences'].values,
                                           tags=test_data['Tags'].values)
    


if __name__ == '__main__':
    main()
