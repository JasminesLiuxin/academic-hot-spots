import codecs
import json
import torch
import random
import spacy
import math
import pickle
import os
import numpy as np
from torch.utils.data import Dataset


class Vocab(object):
    def __init__(self, filename):
        self._token2idx = pickle.load(open(filename, "rb"))
        self._idx2token = {v: k for k, v in self._token2idx.items()}

    def size(self):
        return len(self._idx2token)

    def convert_ids_to_tokens(self, x):
        if isinstance(x, list):
            return [self.convert_ids_to_tokens(i) for i in x]
        return self._idx2token[x]

    def convert_tokens_to_ids(self, x):
        if isinstance(x, list):
            return [self.convert_tokens_to_ids(i) for i in x]
        return self._token2idx[x]

class NLP:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None:
            return text
        text = ' '.join(text.split())
        if lower:
            text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)


class S2SDataset(Dataset):
    """Dataset for sequence-to-sequence generative models, i.e., BART"""

    def __init__(self, data_dir, dataset, tokenizer, node_vocab, relation_vocab, num_samples, usage):
        self.data_dir = data_dir
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.node_vocab = node_vocab
        self.relation_vocab = relation_vocab
        self.num_samples = num_samples
        self.usage = usage
        self.input_nodes, self.input_edges, self.input_types,self.node_lists,self.node_lens,self.relations, self.positions = self.prepare_data()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        idx = 0
        return self.input_nodes[idx], self.input_edges[idx], self.input_types[idx],self.node_lists[idx],self.node_lens[idx], self.relations[idx], self.positions[idx]

    def prepare_data(self):
        words=[]
        #读取数据集
        try:
            data = torch.load(os.path.join(self.data_dir, self.dataset, '{}_{}.tar'.format(self.usage, self.num_samples)))
            #这里需要保存一个实体内容id的 [实体数量*每一个实体的长度，内容是id] 和一个标签实体分词后id的键值对  [时间窗口*标签数量*每一个实体的长度，内容是id]
            input_nodes, input_edges, input_types,node_lists,node_lens,  relations, positions= \
                data["input_nodes"], data["input_edges"], data["input_types"],data["node_lists"],data["node_lens"],data["relations"],data["positions"]
        except FileNotFoundError:
            data_file = os.path.join(self.data_dir, self.dataset, '{}all_results_datasets_processed.json'.format(self.usage))
            input_nodes, input_edges, input_types, node_lists, node_lens,  relations, positions=[],[],[],[],[],[],[]
            # 特征中 每一个节点对应的序列的大小node_list   以及每一个词的长度node_len
            # 标签中 每一个节点对应的序列的大小target_set   以及每一个词的长度target_len
            all_data=[]
            with codecs.open(data_file, "r") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    all_data.append(data)
                data=all_data[0]
                #nodes=data["ent2vocab_id"]
                node_list=[] #1040
                node_len=[] #1040
                for ii in data["nodes"]:
                    words.append(ii)
                    node_list_one=self.tokenizer.tokenize(ii)
                    aa=self.tokenizer.convert_tokens_to_ids(node_list_one)
                    node_list.append(aa)
                    node_len.append(len(node_list_one))
                edges = data["edges"]#2*1696
                types = self.relation_vocab.convert_tokens_to_ids(data["types"]) #1696
                rela = self.relation_vocab.convert_tokens_to_ids(data["relations"]) #848
                pos = data["positions"] #1040
                assert len(types) == len(edges[0]), "The length of edges and types should be matched."
                input_nodes.append(self.node_vocab.convert_tokens_to_ids(data["nodes"])) #对应词表中的节点id  1040
                input_edges.append(edges) #2*1696
                input_types.append(types) #1696
                node_lists.append(node_list)  #对应预训练模型中的每一个词的id  1040*XXX
                node_lens.append(node_len) #1040
                relations.append(rela) #848
                positions.append(pos)#对应图中的节点id    1040长度/图中节点数

            data = {"input_nodes": input_nodes, "input_edges": input_edges, "input_types": input_types, "node_lists": node_lists,"node_lens": node_lens,
                   "relations": relations, "positions": positions
                    }
            file = open("2017words.txt", 'w')
            file.write(str(words));


            torch.save(data, os.path.join(self.data_dir, self.dataset, '{}_{}.tar'.format(self.usage, self.num_samples)))

        return input_nodes, input_edges, input_types,node_lists,node_lens, relations, positions
