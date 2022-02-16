from itertools import Counter
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from tracemalloc import Statistic

import numpy as np
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer



class Statistics(object):
    def __init__(vocab, self):
        # vocab 없이 들어온 dataset만으로.. 구축..? 토크나이저 필요함 근데.
        self.vocab = vocab 
        
        
    def idf(self, dataset, eps=1e-3):
        """[summary]

        Args:
            dataset ([type]): [description]
            eps ([type], optional): [description]. Defaults to 1e-3.

        Returns:
            dict: {x: IDF(x)}
        """
        # total number of documents(for training set)
        D = len(dataset)
        
        doc_counter = defaultdict(int)

        for token in self.vocab:
            for doc, _ in dataset:
                doc_tokens = self.vocab(doc)
                
                # 해당 token을 가진 document 갯수 세기
                if token in doc_tokens:
                    doc_counter[token] += 1
        
        stats = {}

        # compute idf
        for token, count in doc_counter.items():
            idx = self.vocab(token) #attgen.py 에서 손 쉽게 stat을 concat시키기위해서 w:stat 아닌 i:stat으로 했음
            stats[idx] = np.log(np.divide(D, count))
            
        return stats

    def unigram(self, dataset, eps=1e-3):
        """unigram stats. 소스코드의 args.meta_iwf == s(x)

        Args:
            dataset (list): [description]
            eps ([type], optional): [description]. Defaults to 1e-3.

        Returns:
            stats(dict): {x : p(x)}
        """
        counter = Counter()
        
        for text, _ in dataset:
            tokens = self.vocab(text)
            counter.update(tokens)

        stats = {}
        total_count = sum(counter.values())

        for word, count in dict(counter).items():
            uni_p = count / total_count
            idx = self.vocab(word) 
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