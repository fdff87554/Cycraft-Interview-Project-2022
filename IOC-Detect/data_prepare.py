import numpy as np
import pandas as pd

# Read the Original Data
train_data_path = '../data/CTI-reports-dataset/data/CTI reports/train_data'
test_data_path = '../data/CTI-reports-dataset/data/CTI reports/test_data'
train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['Words', 'Tags'], 
                         encoding='utf-8', skip_blank_lines=False)
test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['Words', 'Tags'],
                        encoding='utf-8', skip_blank_lines=False)
print('Train Data Shape: ', train_data.shape)
print('Test Data Shape: ', test_data.shape)
print(train_data.head())
print(test_data.head())
print()

train_data_str = []
test_data_str = []
train_datas = np.split(train_data, train_data[train_data.isnull().all(1)].index)
for data in train_datas:
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    word, tag = data['Words'].str.cat(sep=' '), data['Tags'].str.cat(sep=' ')
    train_data_str.append([word, tag])
    
test_datas = np.split(test_data, test_data[test_data.isnull().all(1)].index)
for data in test_datas:
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    word, tag = data['Words'].str.cat(sep=' '), data['Tags'].str.cat(sep=' ')
    test_data_str.append([word, tag])

train_data_str = pd.DataFrame(train_data_str, columns=['Sentences', 'Tags'])
test_data_str = pd.DataFrame(test_data_str, columns=['Sentences', 'Tags'])
print(train_data_str)
print(test_data_str)

train_data_str.to_csv('../data/cti_train_data.csv', index=False)
test_data_str.to_csv('../data/cti_test_data.csv', index=False)
