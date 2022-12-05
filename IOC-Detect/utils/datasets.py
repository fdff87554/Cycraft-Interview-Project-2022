import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import Dataset


class Data2Idx:
    def __init__(self, maps: dict) -> None:
        self.maps = maps

    def __call__(self, data: np.array) -> torch.Tensor:

        return torch.tensor([self.maps[d] for d in data])


class CTIDatasetForBiLSTMCRF(Dataset):
    def __init__(self, sentences: np.array, tags: np.array, sequence_len: int, 
                 transforms=None, target_transforms=None) -> None:
        super(CTIDatasetForBiLSTMCRF, self).__init__()

        self.vocab_size, self.vocab2idx, self.idx2vocab = self.__get_vocabs(sentences, True)
        self.tag_size, self.tag2idx, self.idx2tag = self.__get_vocabs(tags)

        self.sentences, self.tags = self.__data_preprocess(sentences, tags, sequence_len)

        self.transforms = transforms(self.vocab2idx) if transforms else None
        self.target_transforms = target_transforms(self.tag2idx) if target_transforms else None

    def __get_vocabs(self, sentences: np.array, is_sentences: bool=False) -> tuple[int, dict]:
        vocabs = ['<START>', '<STOP>', '<PAD>']

        sentences = sentences.flatten()
        for sentence in sentences:
            words = sentence.split(' ')
            if is_sentences:
                words = [word.lower() for word in words]
            vocabs.extend(words)
        vocabs = list(set(vocabs))
        # map vocab to index
        vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}
        # map index to vocab
        idx2vocab = dict(enumerate(vocabs))

        return len(vocabs), vocab2idx, idx2vocab

    def __data_preprocess(self, sentences: np.array, tags: np.array, sequence_len: int) -> tuple[np.array, np.array]:
        sentences = self.__padding(self.__data_std(sentences, True), sequence_len)
        tags = self.__padding(self.__data_std(tags), sequence_len)

        return sentences, tags

    def __data_std(self, datas: np.array, is_sentences: bool=False) -> np.array:
        outputs = []
        for data in datas:
            if is_sentences:
                data = data.lower()
            outputs.append(np.array(['<START>'] + data.split(' ') + ['<STOP>']))

        return np.array(outputs, dtype=object)

    def __padding(self, datas: np.array, sequence_len: int) -> np.array:
        outputs = []
        for data in datas:
            if len(data) < sequence_len:
                data = np.append(data, ['<PAD>'] * (sequence_len - len(data)))
            outputs.extend(data[i:i+sequence_len] for i in range(len(data)-sequence_len+1))

        return np.array(outputs)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.transforms(self.sentences[idx])
        labels = self.target_transforms(self.tags[idx])
        mask = (features != self.vocab2idx['<PAD>'])

        return features, labels, mask


class CTIDatasetForBiLSTMCRFWithNgram(Dataset):
    def __init__(self, data_path: str, max_seq_len: int, n_gram: int, transforms=None, target_transforms=None) -> None:
        super(CTIDatasetForBiLSTMCRFWithNgram, self).__init__()
        
        data = self._load_data(data_path)
        word_tag_map = self._build_label_map(data)
        self.vocab_size, self.vocab2idx, self.idx2vocab = self._build_maps('vocab', word_tag_map)
        self.tag_size, self.tag2idx, self.idx2tag = self._build_maps('tag', word_tag_map)
        sentences = self._build_sentences(data)
        data = []
        for i in range(n_gram):
            data.extend(self._build_grams(sentences, max_seq_len, i + 1, max_df=0.8))
        del sentences
        
        self.sentences, self.tags = self._build_dataset(data, max_seq_len, word_tag_map)
        del data, word_tag_map
        
        self.transforms = transforms(self.vocab2idx) if transforms else None
        self.target_transforms = target_transforms(self.tag2idx) if target_transforms else None
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        
        return pd.read_csv(data_path, sep='\t', header=None, names=['Word', 'Tag'], 
                           encoding='utf-8', skip_blank_lines=False)
        
    def _build_label_map(self, data: pd.DataFrame) -> dict:
        word_tag_map = {}
        for _, row in data.iterrows():
            word = str(row['Word']).lower()
            tag = row['Tag']
            if word not in word_tag_map:
                word_tag_map[word] = tag
        word_tag_map['<PAD>'] = '<PAD>'
        word_tag_map['<UNK>'] = '<UNK>'
        word_tag_map['<START>'] = '<START>'
        word_tag_map['<STOP>'] = '<STOP>'
        
        return word_tag_map
    
    def _build_maps(self, map_type: str, word_tag_map: dict) -> tuple[int, dict, dict]:
        if map_type == 'vocab':
            data = list(set(word_tag_map.keys()))
        elif map_type == 'tag':
            data = list(set(word_tag_map.values()))
        else:
            raise ValueError('map_type must be vocab or tag')
        data.extend(['<PAD>', '<UNK>', '<START>', '<STOP>'])
        
        size = len(data)
        data2idx = {data[i]: i for i in range(size)}
        idx2data = {i: data[i] for i in range(size)}
        
        return size, data2idx, idx2data
    
    def _build_sentences(self, data: pd.DataFrame) -> list:
        sentences = []
        data = np.split(data, data[data.isnull().all(1)].index)
        for d in data:
            d.dropna(inplace=True)
            d.reset_index(drop=True, inplace=True)
            sentence = ' '.join(d['Word'].values).lower()
            sentences.append(sentence)
            
        return sentences
    
    def _build_grams(self, sentences: list, max_seq_len: int, n_gram: int, analyzer: str='word', max_df: float=0.9) -> list:
        rets = []
        vectorizer = CountVectorizer(strip_accents='unicode',
                                     lowercase=True,
                                     ngram_range=(n_gram, n_gram),
                                     analyzer=analyzer,
                                     max_df=max_df)
        vectorizer.fit_transform(sentences)
        data = vectorizer.get_feature_names_out(sentences)
        for i in range(len(data)):
            if i % (max_seq_len // n_gram) == 0:
                if i != 0:
                    rets.append(' '.join(ret))
                ret = []
            ret.append(data[i])

        return rets
    
    def _build_dataset(self, data: list, max_seq_len: int, word_tag_map: dict) -> tuple[np.array, np.array]:
        sentences = []
        tags = []
        for d in data:
            sentence = d.split()
            if len(sentence) > (max_seq_len - 2):
                sentence = sentence[:(max_seq_len - 2)]
            else:
                sentence.extend(['<PAD>'] * ((max_seq_len - 2) - len(sentence)))
            sentence = ['<START>'] + sentence + ['<STOP>']
            tag = [word_tag_map.get(w, '<UNK>') for w in sentence]
            for i in range(len(sentence)):
                if tag[i] == '<UNK>':
                    sentence[i] = '<UNK>'
            
            sentences.append(np.array(sentence))
            tags.append(np.array(tag))
            
        return np.array(sentences), np.array(tags)
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentence = self.sentences[idx]
        tag = self.tags[idx]
        mask = torch.tensor(sentence != '<PAD>')
        
        if self.transforms:
            sentence = self.transforms(sentence)
        if self.target_transforms:
            tag = self.target_transforms(tag)
            
        return sentence, tag, mask
