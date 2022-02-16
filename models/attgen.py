import os
from unicodedata import bidirectional
from winreg import REG_RESOURCE_REQUIREMENTS_LIST
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embedding import MetaEmbedding
from einops import rearrange, reduce, repeat




class AttGenerator(nn.Module):
    def __init__(self, args):
        """Assess word importance from the distributional signatures of each input example. Use unigram statistics(provably robust to word-substitution perturbations)
        hidden dim 50
        biLSTM
        dropout=0.1
        embedding: fastText
        sent.ev(huffpost, fewrel) -> pre-trained bert emgbeddings(huggingface)
        for FewRel -> augemnt the input of att generator w/ positinal embeddings(zhang et al., 2017)

        Args:
            args ([type]): [description]
        """
        super(AttGenerator, self).__init__()
        self.args = args
        
        # select embedding table (FastText or BERT / default vs. avg vs. cnn vs. idf)
        self.embedding = MetaEmbedding(args)
        
        if args.word_represent == 'default':
            # unigram + entropy^-1
            input_dim = self.embedding.shape[-1] + 1 + 1
        else:
            input_dim = self.embedding.shape[-1]
            
            
        self.lstm = nn.LSTM(input_dim,
                            args.hid_dim,
                            batch_first=True,
                            dropout=0.1,
                            bidirectional=args.bidirectional
                            )
        
        # learnable vector for dot-prod att
        if args.bidirectional:
            self.v = nn.Linear(2 * args.hid_dim, 1, bias=False)
            # self.v = nn.Parameter(torch.tensor(2 * args.hid_dim, dtype=torch.float))
        else:
            self.v = nn.Linear(args.hid_dim, 1, bias=False)
            # self.v = nn.Parameter(torch.tensor(args.hid_dim, dtype=torch.float))
        
        
    def entropy_to_tensor(self, input_txt, stat):
        


        
    def stat_to_tensor(self, input_txt, stat):
        """입력에 concat시킬 수 있는 형태로 stat 변환
        Args:
            input_txt (tensor): (N*K, max-len)
            stat (dict): 
        """
        assert int(input_txt.shape[0]) == len(stat)
        
        stat_for_concat = torch.zeros_like(input_txt, dtype=torch.float)
        
        for i, doc in enumerate(input_txt):
            for j, token_idx in enumerate(doc):
                stat_for_concat[i][j] = stat[token_idx]
                
        return stat_for_concat
        
        # pad_idx 스탯은 어찌 처리할진 일단 무시하자
        # task 단위로 bpp 시킬꺼임 -> (B, ) 제거.


    # x (tensor): (N*K, max-len)
    #         <pad>들은 어떻게하지????????????? data['w_target'] 딕셔너리 도대체 어딨나?????
    #         -> https://github.com/YujiaBao/Distributional-Signatures/blob/master/src/embedding/avg.py
    #         <pad>, <unk> 제외하고 avg embedding 구하기
    #         -> avg emb따로 클래스로 내
    #         -> input representation phi_x위해 EMB클래스 따로 구현해놓음.
    #         gt (N*K, N)
    def get_cls_specific_word_importance(self, x, gt, pad_idx):
        """get t(x)

        Args:
            x (_type_): _description_
            gt (_type_): _description_
            pad_idx (_type_): _description_

        Returns:
            dict: {token : entropy^-1}
        """
        max_len = x.shape[-1]
        N = gt.shape[-1]

        phi_x = self.embedding(x, 'avg')
        w = reg_simple_classify(phi_x, gt)
        
        
        emb = self.embedding(x) #(N*K, max-len, dim)
        prob = F.softmax(emb @ w) #(N*K, max_len, N)
        
        
        gt = repeat(gt, 'nk n -> nk c n', c=max_len)
        
        # masking하고, GT 인덱스에 대한 확률 값을 p(y|x_i)
        
        entropy_stat = {}
        
        for i, (doc, label) in enumerate(zip(x, gt)):
            # key=token, value = reg prob of token (get from prob literal)
            for j, token in enumerate(doc):
                if token != pad_idx:
                    pred = prob[i, j, :] #(N, )
                    entropy_stat[token] = -torch.inverse(torch.sum(pred * torch.log2(pred))) # entropy^-1

                    
        return entropy_stat

        
    # https://github.dev/YujiaBao/Distributional-Signatures/blob/master/src/embedding/meta.py
    @staticmethod
    def reg_simple_classify(input_represent, gt, l2_factor=1.0):
        """A.1 Regularized Linear classifier

        Args:
            input_represent (_type_): _description_
            gt (_type_): _description_
            l2_factor (float, optional): L2 Norm weighting factor. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        _, D = input_represent.shape
        _, N = gt.shape
        
        w = rand_init(D, N)
        b = rand_init(N, requires_grad=False)
        
        opt = torch.optim.Adam([w, b], lr=0.1)
        
        pred = x @ w + b
        
        # grad norm threshold
        ths = 0.1 
        
        while True:
            opt.zero_grad()
            
            wnorm = torch.linalg.norm(w)
            
            loss = F.cross_entropy(pred, gt) + l2_factor * wnorm  # L2 Norm penalty
            loss.backward()
            
            wgrad = w.grad.data.item()
            
            if torch.linalg.norm(wgrad) < ths:
                break
        
            opt.step()
        
        return w
    
    
    
 
    def forward(self, x, input_len, gt, pad_idx, src_stat=None):
        """[summary]

        Args:
            x (tensor): (N*K, max-len)
            input_len (list): length = N*K 
            src_stat (dict, optional):

        Returns:
            [type]: [description]
        """
        
        # naive embedding(FastText)
        emb = self.embedding(x)
        
        mask = (x == pad_idx)
        
        # concat stat 
        assert src_stat is not None, "source pool statistics must be given to the AttGenerator"
        
        # list for input
        rnn_input = []
        rnn_input.append(emb)
        
        # unigram statistics from source pool - (1)
        if src_stat is not None:
            u = self.stat_to_tensor(x, src_stat['uni'])
            rnn_input.append(u)

            # inverse of entropy statistics - (2)
            if 'ent' in self.args.stat_type:
                entropy_stat = self.get_cls_specific_word_importance(x, gt, pad_idx)
                t = self.stat_to_tensor(x, gt, entropy_stat)
                rnn_input.append(t)
                
        # default - concat [emb, u, t]
        rnn_input = torch.cat(rnn_input, dim=-1)
            
        # pass biLSTM
        rnn_input = pack_padded_sequence(rnn_input, batch_first=True, lengths=input_len)    
        padded_hs, _ = self.lstm(rnn_input) #(N*K, max-len, 2*hid_dim)
        hs = pad_packed_sequence(padded_hs, batch_first=True) #(N*K, max-len, 2*hid_dim)

        # compute attention score
        scores = self.v(hs) #(N*K, max-len, 1)
        scores = reduce(scores, 'nk l 1 -> nk l')
        scores.masked_fill_(mask, 1e-10) #<pad>위치는 softmax 계산 안하도록
        scores = F.softmax(scores) #(N*K, max-len)
        return scores