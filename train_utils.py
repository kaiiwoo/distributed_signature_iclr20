from tracemalloc import Statistic
from itertools import chain
import torch
import numpy as np

from stats import Statistics


"""[summary]

    Args:
        support = torch.stack(support, dim=0) #(N*K, max-len)
        support_gt = torch.stack(support_gt, dim=0) #(N*K, N)
        query = torch.stack(query, dim=0) #(N*L, max-len)
        query_gt = torch.stack(query_gt, dim=0) #(N*L, N)

    return {
        "support" : (support, length)
        "support_gt" : support_gt,
        "query" : (query, length)
        "query_gt" : query_gt,
        "src_stats" : source_stats,
        "sup_stats" : support_stats
    }
    """
    
def compute_acc(pred, gt):
    n_total = gt.shape[0]
    return torch.sum(torch.argmax(pred, dim=-1) == gt) / n_total * 100
    

    
def train_single_task(model, optim, data):
    model.train()
    optim.zero_grad()
    
    query_pred = model(data)
    loss = F.cross_entropy(query_pred, q_gt)

    acc = compute_acc(query_pred, q_gt)
    
    if loss is not None:
        loss.backward()
    if torch.isnan(l):
        return
    
    optim.step()
    
    return acc 

    
def test_single_task(model, data):
    model.eval()
    
    pred = model(data)
    


def make_batch_for_epoch(args, dataset):
    """[summary]

    Args:
        args ([type]): [description]
        dataset ([type]): [description]

    Returns:
        dict: value is list
    """
    classes = dataset.classes
    vocab = dataset.vocab
    S = Statistics(vocab)
    
    N = args.n_way
    K = args.n_shot
    L = args.n_query
    
    support = []
    support_total_len = []
    support_gt = []

    query = []
    query_total_len = []
    query_gt = []
    
    source_stats = [] #length = B
    support_stats = [] #length = B
    
    for _ in range(args.n_episode):
        sampled_cls = np.random.sample(classes, N)

        # 서포트셋과 소스풀은, 매 에피소드마다 새롭게 쓰여져야 함.
        src_pool = [x for x in dataset if x["label"] not in sampled_cls] 
        longest = torch.max([x["text"].shape[0] for x in dataset if x["label"] in sampled_cls])

        # get statistics. dictionary
        uni_src = S.get_stats(src_pool, stat_type='uni')
        idf_src = S.get_stats(src_pool, stat_type='idf')
        s = {'uni' : uni_src, 'idf' : idf_src}
        source_stats.append(s)

        # shot들 순서는 유지했는가?
        # https://github.com/YujiaBao/Distributional-Signatures/blob/master/src/dataset/parallel_sampler.py line 90 보니까 유지한 듯
        support_set = []
        support_length = []
        support_label = []
        
        query_set = []
        query_length = []
        query_label = []
        
        # 한 에피소드 내 N*K개 doc(or sent)에 등장하는 모든 단어들에 대한 Uni / idf stat 캐싱할 dict
        uni_stat_dict = {}
        idf_stat_dict = {}
        
        # prepare support & query set N번 반복하며
        for i, label in enumerate(sampled_cls):
            gt = torch.zeros(N, dtype=torch.float)
            gt[i] += 1.0
            
            target_data = [x for x in dataset if x["label"] == label]
            
            ########### 나 이거 왜 구한거지.../ 2022.02.14
            # update stat info of the target data
            # uni_tgt = S.get_stats(target_data, stat_type='uni')
            # idf_tgt = S.get_stats(target_data, stat_type='idf')
            
            # doc(sent)에 등장하는 단어들의 stat 추가 및 업데이트 
            # uni_stat_dict.update(uni_tgt)
            # idf_stat_dict.udpate(idf_tgt)
            
            # random sample안하고, N_shot & N_query의 크기만큼 인덱싱해서 support & query 구축함. -> 소스코드 구현 방식 따라하기...
            s_data = [pad_sent(x["text"], vocab['<PAD>'], max_len=longest) for x in target_data[:K]] #(K * max-len)
            s_data = torch.stack(s_data, dim=0) #(K, max-len)
            s_len = [x["length"] for x in target_data[:K]] #length = K
            s_label = gt.repeat(K) #(K, N)

            q_data = [pad_sent(x["text"], vocab['<PAD>'], max_len=longest) for x in target_data[K : K+L]] #(L * max-len)
            q_data = torch.stack(q_data, dim=0) #(L, max-len)
            q_len = [x["length"] for x in target_data[K : K+L]] #length = L
            q_label = gt.repeat(L) #(L, N)
            
            support_set.append(convert_tensor(s_data))
            support_length.append(s_len)
            support_label.append(convert_tensor(s_label))
            
            query_set.append(convert_tensor(q_data))
            query_length.append(q_len)
            query_label.append(convert_tensor(q_label))
            
        # N*K개 doc(sent)에 대한 stat 구분을 위한 meta-dict
        # sup_stat_dict = {'uni': uni_stat_dict, 'idf': idf_stat_dict}
        
        # support_stats.append(sup_stat_dict)
        
        # reduce (N, K) -> N*K 
        support_set = torch.cat(support_set, dim=0) #(N*K, max-len)
        support_length = list(chain.from_iterable(support_length)) #length k list X N -> length = N * K (single list)
        support_label = torch.cat(support_label, dim=0) #(N*K, N)
        
        query_set = torch.cat(query_set, dim=0) #(N*L, max-len)
        query_length = list(chain.from_iterable(query_length)) #(encapsulated list) length k list X N -> length = N * K (single list)
        query_label = torch.cat(query_label, dim=0) #(N*L, N)
        
        # save to total data cache
        support.append(support_set) 
        support_total_len.append(support_length) #(B, N*K) (type:list)
        support_gt.append(support_label) 

        query.append(query_set) 
        query_total_len.append(query_length) # (B, N*K) (type:list)
        query_gt.append(query_label) 
        

    # Get Fianl dataset for single epoch
    support = torch.stack(support, dim=0) #(B, N*K, max-len)
    support_gt = torch.stack(support_gt, dim=0) #(B, N*K, N)
    query = torch.stack(query, dim=0) #(B, N*L, max-len)
    query_gt = torch.stack(query_gt, dim=0) #(B, N*L, N)

    return {
        "support" : (support, support_total_len),
        "support_gt" : support_gt,
        "query" : (query, support_total_len),
        "query_gt" : query_gt,
        "src_stats" : source_stats,
        "sup_stats" : support_stats ############### dummy list임.
    }