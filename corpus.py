from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg
from CONFIGS import *
from tqdm import tqdm
import re
import nltk
import json
import pickle
import numpy as np
nltk.download("gutenberg")
sw = nltk.corpus.stopwords.words('english')


class Corpus:
    def __init__(self, init=False) -> None:
        self.idx_word = {}
        self.word_idx = {}
        self.idx_prob = {}
        self.corpus_pair = []
        if init:
            self.read_vocab()
        pass

    def read_vocab(self):
        with open("word_idx.json", 'r') as f:
            self.word_idx = json.load(f)
        with open("idx_word.json", 'r') as f:
            self.idx_word = json.load(f)
        with open("corpus_pair.pkl", 'rb') as f:
            self.corpus_pair = pickle.load(f)
        with open("idx_prob.json", 'r') as f:
            self.idx_prob = json.load(f)
        return

    @staticmethod
    def context_word_pair(sent):
        if(len(sent) < (2*WINDOW_LENGTH+1)):
            return []
        pairs = []
        for i in range(WINDOW_LENGTH, len(sent)-WINDOW_LENGTH):
            center_word = sent[i]
            center_word = re.sub(r'[^a-zA-Z]', '', center_word)
            center_word = center_word.lower()
            if len(center_word) <= 1 or center_word in sw:
                continue
            context = sent[i-WINDOW_LENGTH:i]+sent[i+1:i+WINDOW_LENGTH+1]
            context = [word.lower() for word in context]
            filtered_context = [re.sub(r'[^a-zA-Z]', '', word)
                                for word in context]
            filtered_context = [word for word in filtered_context if word.isalnum(
            ) == True and word not in sw and len(word) > 1]
            if len(filtered_context) == 0:
                continue
                print(center_word, filtered_context)
            pairs.append((center_word, filtered_context))
        return pairs

    @staticmethod
    def get_sentences(fileid=None):
        corpus = []
        if fileid == None:
            for f in gutenberg.fileids():
                corpus += gutenberg.sents(f)
        else:
            for f in fileid:
                corpus += gutenberg.sents(f)
        return corpus

    def get_vocab(self, fileid=None):
        id = 0
        print("Getting Vocab")
        # if fileid == None:
        #     for f in tqdm(gutenberg.fileids()):
        #         sentences = gutenberg.sents(f)
        #         words = set().union(*sentences)
        #         for word in words:
        #             word = word.lower()
        #             word = re.sub(r'[^a-zA-Z]', '', word)
        #             if word in sw or word.isalnum() == False or len(word) <= 1:
        #                 continue
        #             if word not in word_idx:
        #                 word_idx[word] = id
        #                 idx_word[id] = word
        #                 id += 1
        # else:
        #     for f in fileid:
        #         sentences = gutenberg.sents(f)
        #         words = set().union(*sentences)
        #         for word in words:
        #             word = word.lower()
        #             word = re.sub(r'[^a-zA-Z]', '', word)
        #             if word in sw or word.isalnum() == False or len(word) <= 1:
        #                 continue
        #             if word not in word_idx:
        #                 word_idx[word] = id
        #                 idx_word[id] = word
        #                 id += 1
        with open('analogy_corpus.txt', 'r', encoding='utf-8') as file:
            data = file.readlines()
            for line in data:
                for word in line.split():
                    word = word.lower()
                    word = re.sub(r'[^a-zA-Z]', '', word)
                    if word in sw or word.isalnum() == False or len(word) <= 1:
                        continue
                    if word not in self.word_idx:
                        self.word_idx[word] = id
                        self.idx_word[id] = word
                        id += 1
        self.word_idx[UNK_TAG] = id
        self.idx_word[id] = UNK_TAG
        with open("word_idx.json", 'w') as f:
            json.dump(self.word_idx, f)
        with open("idx_word.json", 'w') as f:
            json.dump(self.idx_word, f)
        return

    def get_cbow_pairs(self, fileid=None):
        corpus = []
        # if fileid == None:
        #     for f in tqdm(gutenberg.fileids()):
        #         sentences = gutenberg.sents(f)
        #         for sentence in sentences:
        #             pairs = Corpus.context_word_pair(sentence)
        #             corpus.extend(pairs)
        # else:
        #     for f in tqdm(fileid):
        #         sentences = gutenberg.sents(f)
        #         for sentence in sentences:
        #             pairs = Corpus.context_word_pair(sentence)
        #             corpus.extend(pairs)
        # return corpus
        data = []
        with open('analogy_corpus.txt', 'r', encoding='utf-8') as file:
            data = file.readlines()
            # print(data)
            for line in data:
                pairs = Corpus.context_word_pair(line.split())
                self.corpus_pair.extend(pairs)
        with open('corpus_pair.pkl', 'wb') as f:
            pickle.dump(self.corpus_pair, f)
        return corpus

    @staticmethod
    def get_analogy_pairs():
        corpus = []
        data = ""
        with open('analogy_corpus.txt', 'r', encoding='utf-8') as file:
            data = file.read().split('\n')
        filtered_data = [line.split() for line in data]
        for sentence in filtered_data:
            pairs = Corpus.context_word_pair(sentence)
            corpus.extend(pairs)
        return corpus

    @staticmethod
    def get_probabilities():
        with open('analogy_corpus.txt', 'r', encoding='utf-8') as file:
            vocab = file.read().split('\n')
            data = [line.split() for line in vocab]
            words = [re.sub(r'[^a-zA-Z]', '', word.lower())
                     for sentence in data for word in sentence if (re.sub(r'[^a-zA-Z]', '', word.lower()) not in sw and re.sub(r'[^a-zA-Z]', '', word.lower()).isalnum() != False and len(re.sub(r'[^a-zA-Z]', '', word.lower())) > 1)]
        freq_map = {}
        for word in words:
            if word not in freq_map:
                freq_map[word] = 1
            else:
                freq_map[word] += 1

        total = sum([(freq_map[word]**(3/4)) for word in freq_map])

        word_idx = {}
        with open("word_idx.json", 'r') as f:
            word_idx = json.load(f)
        idx_prob = {}
        for word in freq_map:
            idx_prob[word_idx[word]] = (freq_map[word]**(3/4))/total
        with open("idx_prob.json", 'w') as f:
            json.dump(idx_prob, f)

    def get_samples(self, word, k):
        context = (np.random.choice(list(self.idx_prob.keys()),
                                    p=list(self.idx_prob.values())) for _ in range(k))
        idxs = [self.word_idx[word]] + [idx for idx in context]
        return np.array(idxs, dtype='int')


if __name__ == "__main__":
    corpus = Corpus()
    corpus.get_vocab()
    corpus.get_cbow_pairs()
    corpus.get_probabilities()
    # data = corpus.get_analogy_pairs()
    # print(len(data), data[:3])
    # print(len(word_idx))
