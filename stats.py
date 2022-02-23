from collections import defaultdict, Counter
from multiprocessing.sharedctypes import Value

import torch
import numpy as np
from math import log1p
from torchtext.data.utils import get_tokenizer



class Statistics(object):
    def __init__(self, vocab):
        # vocab 없이 들어온 dataset만으로.. 구축..? 토크나이저 필요함 근데.
        self.vocab = vocab 
        
        
    def idf(self, dataset, eps=1e-3):
        """Inverse document frequency

        Args:
            dataset (list of dict):  return {
            "tokenized": self.token_indexing(self.data[idx]["text"]),
            "label": self.data[idx]["label"],
            "length": length,
            "vocab_size": len(self.vocab)
        }
            eps (float, optional): Defaults to 1e-3.

        Returns:
            dict: {x: IDF(x)}
        """
        # total number of documents(for source pool)
        D = len(dataset)
        counter = Counter()
        
        # count token via lookup_tokens
        for data in dataset:
            tokens = self.vocab.lookup_tokens(data["tokenized"])
            counter.update(tokens)
            
        
        # 맘에 안들어도 먼저 돌아가게,,
        def get_idf(word):
            count = 0
            for doc in [x['tokenized'] for x in dataset]:
                if self.vocab([word])[0] in doc:
                    count += 1

            return log1p(D / eps + count)
            
        idf = {}
        for token in dict(counter).keys():
            idf[token] = get_idf(token)
        
        return idf
            
            
    def unigram(self, dataset, eps=1e-3):
        """unigram stats. 소스코드의 args.meta_iwf == s(x)

        Args:
            dataset (list of dict): 
            eps (float, optional): Defaults to 1e-3.

        Returns:~
            stats(dict): {x : p(x)}
        """
        counter = Counter()
        
        # count token via lookup_tokens
        for data in dataset:
            tokens = self.vocab.lookup_tokens(data["tokenized"])
            counter.update(tokens)

        stats = {}
        total_count = sum(counter.values())
        for word, count in dict(counter).items():
            uni_p = count / total_count
            # input: list / output : list
            idx = self.vocab([word])[0]
            stats[idx] = eps / (eps +  uni_p)
            
        return stats
        

    def entropy(self, word_pred):
        """eq.(2)
            소스코드의 args.meta_w_target -> args.meta_target_entropy
            args.meta_target_entropy == False : https://github.com/YujiaBao/Distributional-Signatures/blob/c613ed070af3e7ae4967b9942fde16864af28cde/src/embedding/meta.py#L200

        Args:
            word_pred (tensor): (n_words, n_classes)

        Returns:
            cond_h: (n_words, )
        """
        cond_h = - word_pred * torch.log2(word_pred)
        cond_h = torch.sum(cond_h, dim=-1)
        return cond_h
    

    def get_stats(self, dataset, stat_type='uni'):
        if stat_type == 'uni':
            return self.unigram(dataset)
        elif stat_type == 'idf':
            return self.idf(dataset)
        else:
            raise ValueError