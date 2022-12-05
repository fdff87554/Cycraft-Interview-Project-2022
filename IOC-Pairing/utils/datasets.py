import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en')


class DependancyParserDataset(Dataset):
    def __init__(self, sentences: np.array, tags: np.array, transforms=None, target_transforms=None) -> None:
        super(DependancyParserDataset, self).__init__()

        self.sentences, self.tags = self.__data_preprocess(sentences, tags)

        self.transforms = transforms or None
        self.target_transforms = target_transforms or None
        
    def __data_preprocess(self, sentences: np.array, tags: np.array) -> tuple[np.array, np.array]:
        fin_tags = []
        for sentence, tag in zip(sentences, tags):
            doc = nlp(sentence)
            tag = tag.split(' ')
            ner = [token.ner for sent in doc.sentences for token in sent.tokens]

            use_ner = False
            for i in range(len(tag)):
                if tag[i] == 'O':
                    if ner[i] == 'O' or 'S' in ner[i][0] or 'E' in ner[i][0]:
                        if use_ner:
                            tag[i] = ner[i]
                        use_ner = False
                    elif 'B' in ner[i][0]:
                        tag[i] = ner[i]
                        use_ner = True
                    elif 'I' in ner[i][0]:
                        if use_ner:
                            tag[i] = ner[i]
                    else:
                        use_ner = False
                else:
                    use_ner = False
            fin_tags.append(' '.join(tag))
        
        return sentences, np.array(fin_tags)
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences[index]
        tag = self.tags[index]

        if self.transforms:
            sentence = self.transforms(sentence)
        if self.target_transforms:
            tag = self.target_transforms(tag)

        return sentence, tag
