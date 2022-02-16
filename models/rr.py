import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import reduce, rearrange

from models.fullmodel import FullModel


class RidgeRegressor(FullModel):
    def __init__(self, args) -> None:
        """
            L2 distance based - quickly learns to make predictions after seeing a few examples
        """
        super(RidgeRegressor, self).__init__()
        self.just_emb = nn.Embedding(args.num_vocab, args.emb_dim)
        self.att = AttGenerator(args)
        self.linear = nn.Linear(args.hid_dim, args.num_classes, bias=False)
        
        # meta parameter in log space 
        # 초기화 어케하는 지 정보없음
        self.lamda = nn.Parameter(torch.tensor(1), dtype=torch.float)
        
        # meta parameters 
        self.a = nn.Parameter(torch.tensor(1), dtype=torch.float)
        self.b = nn.Parameter(torch.tensor(1), dtype=torch.float)
        
    
    def reg_sqr_loss(self, x, w, y):
        return torch.sum((x @ w - y)**2) - self.lamda * torch.sum(w**2)
        


    def compute_w(self, x, y):
        """[summary]

        Args:
            x ([type]): (N*K, hid_dim)
            y ([type]): (N*K, N)

        Returns:
            [type]: (N*K, )
        """
        NK = x.shape[0]
        inv_term = torch.inverse((x @ x.t() + self.lamda * torch.eyes(NK, dtype=torch.float)))
        w = x.t() @ inv_term @ y
        return w
    
    
    def pass_support(self, support, src_stat=None):
        """it weight via training support set

        Args:
            support (_type_): _description_
            src_stat (_type_, optional): _description_. Defaults to None.

        Returns:
            w: (dim, n_classes)
        """
        sup, sup_gt = support 
        text, length = sup[0], sup[1]
        
        # construct representiations eq.(4)
        # normalize안하고 그냥 가중합????
        emb = self.just_emb(text) #(N*K, max-len, emb_dim)
        
        pad_idx = vocab['<PAD>']
        
        att_scores = self.att(text, length, sup_gt, pad_idx, src_stat=src_stat) #(N*K, max-len)
        
        phi_x = torch.sum(emb * att_scores, dim=1) #(N*K, 1, emb_dim)
        phi_x = rearrange(phi_x, 'b 1 d -> b d')

        w = self.compute_w(phi_x, sup_gt)
        return w

        
    def pass_query(self, query, w, src_stat=None):
        q, q_gt = query 
        text, length = q[0], q[1]
        
        emb = self.just_emb(text) #(N*K, max-len, emb_dim)
        
        pad_idx = vocab['<PAD>']
        
        att_scores = self.att(text, length, q_gt, pad_idx, src_stat=src_stat) #(N*K, max-len)
        
        phi_x = torch.sum(emb * att_scores, dim=1) #(N*K, 1, emb_dim)
        phi_x = rearrange(phi_x, 'b 1 d -> b d')

        # inference on query set
        pred = self.a * (phi_x @ w) + self.b
        return pred
        
        
    def forward(self, data):
        """사실상 찐 학습모듈

        Args:
            data (tuple): data

        Returns:
            tensor: prediction. (N*L, n_classes)
        """
        s, s_gt, q, q_gt , src_st, sup_st = data

        support = (s, s_gt)
        query = (q, q_gt)
        
        w = self.pass_support(support, src_st)
        return self.pass_query(query, w, src_st)
        

        
            
        