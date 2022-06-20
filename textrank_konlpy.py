import numpy as np
import pandas as pd
import re
import math
import networkx as nx

from konlpy.tag import Mecab
from hanspell import spell_checker
from tqdm import tqdm


def similarity(a, b):
    try:
        n = len(a.intersection(b))
        return n / float(len(a) + len(b) - n)
    except:
        return 0

class Textrank:
    def __init__(self, **kargs):
        self.dictCount = {}
        self.tag_sen = []
        self.dictBicount={}
        self.graph = None
        self.ratio = 0.666
        self.max_sen = 5
        self.threshold = 0.005

    def load_sentences(self, sentences, tokenizer):
        stopwords = set([('있','VV'),('하','VV'),('되','VV')])
        for sent in sentences.split('.'):
            temp_lst = []
            temp = tokenizer.pos(sent)
            for i in temp:
                if (i not in stopwords) and (i[1] in ('NNG', 'NNP', 'VV', 'VA')):
                    temp_lst.append(i)
            self.tag_sen.append(set(temp_lst))
            self.dictCount[len(self.dictCount)] = sent
        
        for i in range(len(self.dictCount)):
            for j in range(i+1, len(self.dictCount)):
                s = similarity(self.tag_sen[i], self.tag_sen[j])
                if s < self.threshold:
                    continue
                self.dictBicount[i, j] = s

    def build_graph(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.dictCount.keys())
        for (a, b), n in self.dictBicount.items():
            self.graph.add_edge(a, b, weight=n*1.0 + (1-1.0))

    def pagerank(self):
        return nx.pagerank(self.graph, weight='weight')
    


    def new_line(self):
        rank = self.pagerank()
        ks = sorted(rank, key=rank.get, reverse=True)
        score = int(len(rank)*self.ratio)

        if score < self.max_sen :
            score = int(len(rank)*self.ratio)
        elif score >= self.max_sen:
            score = self.max_sen
        else:
            pass

        if score == 0:
            score = len(rank)

        ks = ks[:score]
        new_text = '.'.join(map(lambda k: self.dictCount[k], sorted(ks)))
        return new_text

    def spell_check(text):
        if len(text) >= 500:
            pass
        else:
            text = spell_checker.check(text).as_dict()["checked"]
        return text

def sentence_extraction(sentences):
    tr = Textrank()
    tokenizer = Mecab()
    tr.load_sentences(sentences, tokenizer)
    tr.build_graph()
    rank = tr.pagerank()
    result = tr.new_line()

    if result == "":
        result = sentences
    
    result = tr.spell_check(result)
    return result