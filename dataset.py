import json
from collections import Counter, OrderedDict
import spacy
from tqdm import tqdm


# Create a blank Tokenizer with just the English vocab
from torchtext.vocab import vocab
from torch.utils.data import Dataset

spacy_en = spacy.load('en_core_web_sm')

# class for construting vocabulary
class Vocabulary(object):
    def __init__(self, args):
        self.args = args
        self.counter = Counter()
        self.vocab = None

        self.update_counter(args.data_path)
        
        # construct vocab
        # https://pytorch.org/text/stable/vocab.html
        sorted_by_freq_tuples = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        
        self.vocab = vocab(ordered_dict, min_freq=5)
        self.vocab.insert_token('<UNK>', 0)
        self.vocab.insert_token('<PAD>', 1) 
        self.vocab.set_default_index(self.vocab['<UNK>']) #make default index same as index of unk_token
        print(self.vocab['out of vocab'] is self.vocab['UNK']) # return True
        print(f"사전크기:{len(self.vocab)}")
        print(f"Vocabulary for {args.data_path} has constructed!")
        import pickle
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
        
    def tokenize(self, s):
        # BPE
        if self.args.use_bert:
            pass
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # return tokenizer.encode(s, add_special_tokens=False)
        # word Lv. standard tokenizer
        else:
            return [tok.text for tok in spacy_en.tokenizer(s)]
        
    def update_counter(self, datapath):
        if len(datapath.split('/')) == 4:
            name = datapath.split('/')[2]
        elif len(datapath.split('/')) == 3:
            name = datapath.split('/')[1]
            
        assert isinstance(name, str), "name must be a string type"
        
        if name.lower() == "huffpost":
            with open(datapath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="constructing vocab"):
                    obj = json.loads(line)
                    self.counter.update(self.tokenize(obj["headline"]))
                    
                    

# num of labels for train / val / test
class TextClassificationData(Dataset):
    def __init__(self, args, vocab, classes:list) -> None:
        """TextClassificaiton Dataset

        Args:
            args ([type]): argument
            classes (list): train / val / test로 한정시킬 클래스 리스트
        """
        super(TextClassificationData, self).__init__()
        self.args = args

        self.data = []
        self.classes = classes
        self.stats = {}
        
        self.vocab = vocab

        # parse data
        self.parse_data(args.data_path)
        
    # word to index & tokenize
    def token_indexing(self, s):
        # BPE
        if self.args.use_bert:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            return tokenizer.encode(s, add_special_tokens=False)
        # word Lv. standard tokenizer
        else:
            tokens = [tok.text for tok in spacy_en.tokenizer(s)]
            return self.vocab(tokens)

    # Data paser
    def parse_data(self, datapath):
        if len(datapath.split('/')) == 4:
            name = datapath.split('/')[2]
        elif len(datapath.split('/')) == 3:
            name = datapath.split('/')[1]
            
        assert isinstance(name, str), "name must be a string type"
        
        if name.lower() == "huffpost":
            with open(datapath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="caching data for given class list.."):
                    obj = json.loads(line)
                    
                    # self.classes에 속한 샘플들만 데이터 취득
                    if obj["category"] in self.classes:
                        self.data.append(
                                {
                                    "text": obj["headline"],
                                    "label" : obj["category"]
                                }
                            )
                    else:
                        continue
                        

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                "text" : [ ],
                "label" : str,
                "length" : int,
            }
        """
        ids = self.token_indexing(self.data[idx]["text"])
        if isinstance(ids, list):
            length = len(ids)
        elif isinstance(ids, torch.tensor):
            length = int(ids.shape[0])
        
        return {
            "tokenized": self.token_indexing(self.data[idx]["text"]),
            "label": self.data[idx]["label"],
            "length": length,
            "vocab_size": len(self.vocab)
        }

    def __len__(self):
        return len(self.data)



