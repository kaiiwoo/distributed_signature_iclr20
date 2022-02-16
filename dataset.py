import json
from collections import Counter, OrderedDict
from re import S
import spacy
from spacy import get_tokenizer
from tqdm import tqdm

from torch.utils.data import Dataset



# num of labels for train / val / test
class TextClassificationData(Dataset):
    def __init__(self, args, classes:list) -> None:
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
        
        self.counter = Counter()
        self.vocab = None
        
        # BPE
        if args.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # word Lv. 
        else:
            self.tokenizer = get_tokenizer('spacy', 'en_core_web_sm')
        
        # parse data
        self.parser(args.datapath)
        
        # construct vocab
        sorted_by_freq_tuples = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.vocab = vocab(ordered_dict)
        self.vocab.append_token('<PAD>') #맨 마지막 순번으로 추가 예상

    # word to index & tokenize
    def tokenize(self, s):
        if self.args.use_bert:
            return self.tokenizer.encode(s, add_special_tokens=False)
        else:
            tokens = self.tokenizer(s)
            return self.vocab(tokens)

    # Data paser
    def parser(self, datapath):
        name = datapath.split('/')[0]
        assert isinstance(name, str), "name must be a string type"
        
        if name == "huffpost":
            with open(datapath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="caching data for given class list.."):
                    obj = json.loads(line)
                    
                    # 단어사전 구축용
                    self.counter.update(self.tokenizer(obj["headline"]))
                    
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
        ids = self.tokenize(self.data[idx]["text"])
        if isinstance(ids, list):
            length = len(ids)
        elif isinstance(ids, torch.tensor):
            length = int(ids.shape[0])
        
        
        return {
            "text": self.tokenize(self.data[idx]["text"]),
            "label": self.data[idx]["label"],
            "length": length
        }

    def __len__(self):
        return len(self.data)



