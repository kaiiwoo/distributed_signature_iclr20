import json
import torch
import numpy as np
from tqdm import tqdm

from dataset import TextClassificationData, Vocabulary

def label_extractor(datapath):
    labels = []
    
    if len(datapath.split('/')) == 4:
        name = datapath.split('/')[2]
    elif len(datapath.split('/')) == 3:
        name = datapath.split('/')[1]
    
    if name.lower() == "huffpost":
        with open(datapath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="caching all objects.."):
                obj = json.loads(line)
                labels.append(obj["category"])
                
    return labels


def pad_sent(sent, pad_idx, max_len:int):
    if isinstance(sent, list):
        assert len(sent) <= max_len
        sent += [pad_idx] * int(max_len - len(sent))

    elif isinstance(sent, torch.tensor):
        if len(sent.shape) == 1:
            assert sent.shape[0] <= max_len
            pads = torch.tensor([pad_idx] * int(max_len - len(sent)))
            sent = torch.cat([sent, pads], dim=-1)
            
        elif len(sent.shape) == 2:
            assert sent.shape[-1] <= max_len
            for s in sent:
                pads = torch.tensor([pad_idx] * int(max_len - len(s)))
                s = torch.cat([s, pads], dim=-1)
        
    return sent


def class_splitter(args):
    n_train, n_val, n_test = args.label_split[0], args.label_split[1], args.label_split[2]
    all_cls = label_extractor(args.data_path)
    
    # A.4 Datasets
    print(f"data split: {args.split_type}")
    if args.split_type == "easy":
        all_cls = list(set(all_cls))
        np.random.permutation(all_cls)
        
    # 내림차순으로 정렬된 {카테고리:카운트}
    elif args.split_type == "hard":
        sorted_dict = dict(Counter(all_cls))
        all_cls = list(sorted_dict.keys())

    # list of splited all_cls
    train_cls = all_cls[:n_train]
    val_cls = all_cls[n_train : n_train + n_val]
    test_cls = all_cls[n_train + n_val : n_train + n_val + n_test]
    
    assert len(all_cls) == sum([len(train_cls), len(val_cls), len(test_cls)])

    return {'train_cls': train_cls,
            'val_cls': val_cls,
            'test_cls': test_cls, 
            }

def get_dataset(args):
    cls_dict = class_splitter(args)
    
    # construct vocab from totaldataset. 
    # 원래 학습데이터로만 vocab 구축하는게 맞는데, 원본 코드 보니까 전체 doc가지고 만들어서 그냥 따라간다,,
    vocab = Vocabulary(args).vocab
    
    train_set = TextClassificationData(args, vocab,  classes=cls_dict['train_cls'])
    val_set = TextClassificationData(args, vocab,  classes=cls_dict['val_cls'])
    test_set = TextClassificationData(args, vocab, classes=cls_dict['test_cls'])

    return (train_set, val_set, test_set, vocab)