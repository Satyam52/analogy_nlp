import os
import numpy as np
from CONFIGS import *
from corpus import Corpus
from tqdm import tqdm
import random
import time
random.seed(SEED)
np.random.seed(SEED)


def sig(x):
    return 1/(1 + np.exp(-x))


class CBOW:
    def __init__(self, corpus) -> None:
        self.word_idx, self.idx_word = corpus.word_idx, corpus.idx_word
        self.vocab_size = len(self.word_idx)
        weights = os.listdir('weights/')
        for i in range(50, 0, -1):
            weight_file_w1 = f'w1_{i}.npy'
            weight_file_w2 = f'w2_{i}.npy'
            if weight_file_w1 in weights and weight_file_w2 in weights:
                self.w1 = np.load('weights/'+weight_file_w1)
                self.w2 = np.load('weights/'+weight_file_w2)
                self.epoch = i
                print(f'Loaded weights from {i}th epoch....')
                return
        self.w1 = self.glorot_init(self.vocab_size, EMBEDDING_SIZE)
        self.w2 = self.glorot_init(EMBEDDING_SIZE, self.vocab_size)
        self.sample_mask = np.zeros((1, K+1))
        self.sample_mask[0] = 1

    def glorot_init(self, d1, d2):
        std_dev = np.sqrt(2.0/(d1+d2))
        w = np.random.normal(0, std_dev, size=(d1, d2))
        return w

    def forward(self, X, Y):
        '''
        Inputs are
        X: B x V
        w1: V x D
        w2: D x V
        '''
        # residuals = {}
        # residuals['hidden'] = X @ self.w1  # B x D = B x V @ V x D
        # # B x V = B x D @ D x V
        # residuals['output'] = residuals['hidden'] @ self.w2
        # residuals['softmax'] = self.softmax(residuals['output'])  # B x V
        residuals = {}
        X = X[:, None].T
        residuals['hidden'] = X @ self.w1  # 1 x D =  1 x V @ V x D
        # 1 X (K+1)  = 1 x D @ D x (K+1)
        residuals['output'] = residuals['hidden'] @ self.w2[:, Y]
        residuals['output'] = sig(residuals['output'])
        return residuals

    def backward(self, residuals, X, Y):
        # grad_output = -(residuals['softmax'] - Y)  # B x V = B x V - B x V
        # grad_w2 = residuals['hidden'].T @ grad_output  # D x V = D x B @ B x V
        # grad_hidden = grad_output @ self.w2.T  # B x D = B x V @ V x D
        # grad_w1 = X.T @ grad_hidden  # V x D = V x B @ B x D
        # if np.sum(grad_w1) == 0 or np.sum(grad_w2) == 0:
        #     pass
        #     #print("ZERO GRAD")
        # self.w2 -= ALPHA * grad_w2
        # self.w1 -= ALPHA * grad_w1
        # loss = self.cross_entropy_loss(Y, residuals['softmax'])
        X = X[:, None]
        # 1 x (K+1) = 1 x (K+1) - 1 x (K+1)
        grad_output = residuals['output'] - self.sample_mask
        # D x (K+1) = D x 1 @ 1 x (K+1)
        grad_w2 = residuals['hidden'].T @ grad_output
        # 1 x D = 1 x (K+1) @ (K+1) x D  #EH
        grad_hidden = grad_output @ self.w2[:, Y].T
        grad_w1 = X @ grad_hidden  # V x D = V x B @ B x D
        if np.sum(grad_w1) == 0 or np.sum(grad_w2) == 0:
            print("ZERO GRAD")
        self.w2[:, Y] -= ALPHA * grad_w2
        self.w1 -= ALPHA * grad_w1
        loss = self.cross_entropy_loss(Y, residuals['output'])
        return loss

    def cross_entropy_loss(self, P, Q):
        # Q = np.clip(Q,1e-7,1-1e-7)
        # loss = -np.sum(P * np.log(Q))
        # loss = P*Q
        # return np.mean(-np.log(np.clip(np.sum(loss, axis=1), 1e-7, 1-1e-7)))
        return -np.log(Q[0][0]) - np.sum(np.log(1-Q[0][1:]))

    def softmax(self, X):
        exp_X = np.exp(X)
        sum_exp_X = np.sum(exp_X, 1, keepdims=True)
        softmax_X = exp_X/sum_exp_X
        return softmax_X

    def eval_model(self):
        data = []
        with open('Analogy_dataset.txt', 'r') as f:
            data = f.read().lower().split('\n')
            data = [pair.split() for pair in data]
        data_val = []
        with open('Validation.txt', 'r') as f:
            file = f.read().lower().split('\n')
            data_val = [pair.split() for pair in file]
        print(f'Length of analogy corpus = {len(data)}')
        print(f'Length of validation corpus = {len(data_val)}')
        # print(data[:2],data_val[:2])
        correct = 0
        print("Evaluating Accuracy on analogy set.....")
        for pair in tqdm(data):
            if(len(pair) != 4):
                continue
            a, b, c, d = pair[0], pair[1], pair[2], pair[3]
            x1 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[a] if a in self.word_idx else self.word_idx[UNK_TAG]
            x1[0, idx] = 1
            x1_vec = x1 @ self.w1
            x2 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[b] if b in self.word_idx else self.word_idx[UNK_TAG]
            x2[0, idx] = 1
            x2_vec = x2 @ self.w1
            y1 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[c] if c in self.word_idx else self.word_idx[UNK_TAG]
            y1[0, idx] = 1
            y1_vec = y1 @ self.w1
            y2 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[d] if d in self.word_idx else self.word_idx[UNK_TAG]
            y2[0, idx] = 1
            y2_vec = y2 @ self.w1
            analogy = x2_vec-x1_vec+y1_vec
            analogy_argmax = analogy @ self.w2
            pred_idx = np.argmax(analogy_argmax)
            if self.idx_word[pred_idx] == d:
                correct += 1
        print(f'Accuracy on analogy = {correct/len(data)}')
        correct = 0
        print("Evaluating Accuracy on validation set.....")
        for pair in tqdm(data_val):
            if(len(pair) != 4):
                continue
            a, b, c, d = pair[0], pair[1], pair[2], pair[3]
            x1 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[a] if a in self.word_idx else self.word_idx[UNK_TAG]
            x1[0, idx] = 1
            x1_vec = x1 @ self.w1
            x2 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[b] if b in self.word_idx else self.word_idx[UNK_TAG]
            x2[0, idx] = 1
            x2_vec = x2 @ self.w1
            y1 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[c] if c in self.word_idx else self.word_idx[UNK_TAG]
            y1[0, idx] = 1
            y1_vec = y1 @ self.w1
            y2 = np.zeros((1, self.vocab_size))
            idx = self.word_idx[d] if d in self.word_idx else self.word_idx[UNK_TAG]
            y2[0, idx] = 1
            y2_vec = y2 @ self.w1
            analogy = x2_vec-x1_vec+y1_vec
            analogy_argmax = analogy @ self.w2
            pred_idx = np.argmax(analogy_argmax)
            if self.idx_word[pred_idx] == d:
                correct += 1
        print(f'Accuracy on validation = {correct/len(data_val)}')

    def cosine_similarity(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


if __name__ == "__main__":
    cor = Corpus()
    cor.read_vocab()
    cbow = CBOW(cor)
    word_idx, idx_word = cor.word_idx, cor.idx_word
    vocab_size = len(word_idx)
    print(f'Vocab size = {vocab_size}')
    history = []
    corpus = cor.corpus_pair
    random.shuffle(corpus)
    print("Length of corpus = ", len(corpus))
    print(corpus[:3])
    # corpus = corpus[:500]
    for ep in range(EPOCHS):
        random.shuffle(corpus)
        epoch_loss = 0
        for i in tqdm(range(0, len(corpus))):
            x = np.zeros(vocab_size)
            y = np.zeros(K+1)
            word = corpus[i][0]
            context = corpus[i][1]
            # target_idx = word_idx[word] #if word in word_idx else word_idx[UNK_TAG]
            y = cor.get_samples(word, K)
            cont_cnt = 0
            for k in range(len(context)):
                # if context[k] in word_idx else word_idx[UNK_TAG]
                context_idx = word_idx[context[k]]
                x[context_idx] += 1
                cont_cnt += 1
            x /= cont_cnt
            residuals = cbow.forward(x, y)
            loss = cbow.backward(residuals, x, y)
            epoch_loss += loss
            # print("LOSS")
            # print(loss)

        cbow.eval_model()
        print("\n", epoch_loss)
        history.append(epoch_loss)
        np.save(f"weights/w1_{cbow.epoch+ep}.npy", cbow.w1)
        np.save(f"weights/w2_{cbow.epoch+ep}.npy", cbow.w2)
    print(history)
