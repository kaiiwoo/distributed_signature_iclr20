from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchtext
import numpy as np

from bert import BertRepresentation
from einops import rearrange, reduce, repeat
from torchtext.vocab import Vectors



# embedding 셀렉팅 모듈
# https://github.com/YujiaBao/Distributional-Signatures/blob/master/src/embedding/factory.py

# 논문의 methodology 
# https://github.com/YujiaBao/Distributional-Signatures/blob/c613ed070af3e7ae4967b9942fde16864af28cde/src/embedding/meta.py#L169

# Avg.는 별개의 representation! 어텐션 weighted sum이랑 다른 거임. 구분해서 생각하자. 
class MetaEmbedding(object):
    def __init__(self, args) -> None:
        
        if args.use_bert:
            self.embedding = BertRepresentation(args)
        else: 
            vec = torchtext.vocab.FastText(language='en')
            self.embedding = nn.Embedding.from_pretrained(vec.vectors)
        
    def _shape(self):
        return self.embedding.weight.shape

    def avg_emb(self, x, length):
        """avg emb

        Args:
            x (tensor): (B, max-len, D)
            length (list): length(B)

        Returns:
            holder: (B, D)
        """
        assert int(x.shape[0]) == len(length)

        holder = torch.zeros((x.shape[0], x.shape[2]))
        
        for i, len  in enumerate(length):
            holder[i] =  torch.mean(x[i, :len, :])
            
        return holder
        
        
    def idf_emb(self, x):
        pass

    def cnn_emb(self, x):
        pass
    
    
    def __call__(self, x, represent=None):
        x = self.embedding(x)
        
        if represent == 'default' or None:
            return x
        elif represent == 'avg':
            return self.avg_emb(x)
        elif represent == 'idf':
            return self.idf_emb()
        elif represent == 'cnn':
            return self.cnn_emb()
        else:
            raise ValueError