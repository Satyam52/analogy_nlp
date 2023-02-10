import numpy as np
from CONFIGS import *
from corpus import Corpus
from tqdm import tqdm
import random
import time
random.seed(SEED)
np.random.seed(SEED)
import os

class SkipGram:
    def __init__(self) -> None:
        self.word_idx, self.idx_word = Corpus.get_vocab()
        self.vocab_size = len(self.word_idx)
        weights = os.listdir('weights/')
        for i in range(50,0,-1):
            weight_file_w1 = f'skip_w1_{i}.npy'
            weight_file_w2 = f'skip_w2_{i}.npy'
            if weight_file_w1 in weights and weight_file_w2 in weights:
                self.w1 = np.load('weights/'+weight_file_w1)
                self.w2 = np.load('weights/'+weight_file_w2)
                self.epoch = i
                print(f'Loaded weights from {i}th epoch....')
                return
        self.w1 = self.glorot_init(self.vocab_size,EMBEDDING_SIZE)
        self.w2 = self.glorot_init(EMBEDDING_SIZE,self.vocab_size)

    def glorot_init(self,d1,d2):
        std_dev =  np.sqrt(2.0/(d1+d2))
        w = np.random.normal(0,std_dev,size=(d1,d2))
        return w



    def forward(self,X):
        '''
        Inputs are
        X: B x V
        w1: V x D
        w2: D x V
        '''
        residuals = {}
        residuals['hidden'] = X @ self.w1 # B x D = B x V @ V x D
        residuals['output'] = residuals['hidden'] @ self.w2 # B x V = B x D @ D x V
        residuals['softmax'] = self.softmax(residuals['output']) # B x V
        return residuals
        
    def backward(self,residuals,X,Y):
        grad_output = np.sum(residuals['softmax']-Y,axis=0)
        grad_w2 = residuals['hidden'].T @ grad_output # D x V = D x B @ B x V
        grad_hidden = grad_output @ self.w2.T # B x D = B x V @ V x D
        grad_w1 = X.T @ grad_hidden # V x D = V x B @ B x D
        if np.sum(grad_w1)==0 or np.sum(grad_w2)==0:
            pass
        self.w2 -= ALPHA * grad_w2
        self.w1 -= ALPHA * grad_w1
        loss = self.cross_entropy_loss(Y,residuals['softmax'])
        return loss

    def cross_entropy_loss(self,P,Q):
        #Q = np.clip(Q,1e-7,1-1e-7)
        #loss = -np.sum(P * np.log(Q))
        loss = P*Q
        return np.mean(-np.log(np.clip(np.sum(loss,axis=1),1e-7,1-1e-7)))
        

    def softmax(self,X):
        exp_X = np.exp(X)
        sum_exp_X = np.sum(exp_X,1, keepdims=True)
        softmax_X = exp_X/sum_exp_X
        return softmax_X

    def eval_model(self):
        data = []
        with open('Analogy_dataset.txt','r') as f:
            data = f.read().lower().split('\n')
            data = [pair.split() for pair in data]
        data_val = []
        with open('Validation.txt','r') as f:
            file = f.read().lower().split('\n')
            data_val = [pair.split() for pair in file]
        print(f'Length of analogy corpus = {len(data)}')
        print(f'Length of validation corpus = {len(data_val)}')
        #print(data[:2],data_val[:2])
        correct = 0
        print("Evaluating Accuracy on analogy set.....")
        for pair in tqdm(data):
            if(len(pair)!= 4):
                continue
            a,b,c,d = pair[0],pair[1],pair[2],pair[3]
            x1 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[a] if a in self.word_idx else self.word_idx[UNK_TAG]
            x1[0,idx] = 1
            x1_vec = x1 @ self.w1
            x2 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[b] if b in self.word_idx else self.word_idx[UNK_TAG]
            x2[0,idx] = 1
            x2_vec = x2 @ self.w1
            y1 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[c] if c in self.word_idx else self.word_idx[UNK_TAG]
            y1[0,idx] = 1
            y1_vec = y1 @ self.w1
            y2 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[d] if d in self.word_idx else self.word_idx[UNK_TAG]
            y2[0,idx] = 1
            y2_vec = y2 @ self.w1
            analogy = x2_vec-x1_vec+y1_vec
            analogy_argmax = analogy @ self.w2
            pred_idx = np.argmax(analogy_argmax)
            if self.idx_word[pred_idx] == d:
                correct+=1
        print(f'Accuracy on analogy = {correct/len(data)}')
        correct = 0
        print("Evaluating Accuracy on validation set.....")
        for pair in tqdm(data_val):
            if(len(pair)!= 4):
                continue
            a,b,c,d = pair[0],pair[1],pair[2],pair[3]
            x1 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[a] if a in self.word_idx else self.word_idx[UNK_TAG]
            x1[0,idx] = 1
            x1_vec = x1 @ self.w1
            x2 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[b] if b in self.word_idx else self.word_idx[UNK_TAG]
            x2[0,idx] = 1
            x2_vec = x2 @ self.w1
            y1 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[c] if c in self.word_idx else self.word_idx[UNK_TAG]
            y1[0,idx] = 1
            y1_vec = y1 @ self.w1
            y2 = np.zeros((1,self.vocab_size))
            idx = self.word_idx[d] if d in self.word_idx else self.word_idx[UNK_TAG]
            y2[0,idx] = 1
            y2_vec = y2 @ self.w1
            analogy = x2_vec-x1_vec+y1_vec
            analogy_argmax = analogy @ self.w2
            pred_idx = np.argmax(analogy_argmax)
            if self.idx_word[pred_idx] == d:
                correct+=1
        print(f'Accuracy on validation = {correct/len(data_val)}')

    def cosine_similarity(self,a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))



if __name__ == "__main__":
    SG = SkipGram()
    word_idx, idx_word = SG.word_idx, SG.idx_word
    vocab_size = len(word_idx)
    print(f'Vocab size = {vocab_size}')
    history = []
    corpus = Corpus.get_cbow_pairs()
    corpus += Corpus.get_analogy_pairs()
    random.shuffle(corpus)
    print("Length of corpus = ",len(corpus))
    print(corpus[:3])
    #corpus = corpus[:500]
    for ep in tqdm(range(EPOCHS)):
        random.shuffle(corpus)
        epoch_loss = 0
        for i in tqdm(range(0,len(corpus),BATCH_SIZE)):
            x = np.zeros((BATCH_SIZE,vocab_size))
            y = np.zeros((2*WINDOW_LENGTH, BATCH_SIZE,vocab_size))
            for j in range(i,min(len(corpus),i+BATCH_SIZE)):
                word = corpus[j][0]
                context = corpus[j][1]
                target_idx = word_idx[word] if word in word_idx else word_idx[UNK_TAG]
                x[j-i,target_idx] = 1
                for k in range(len(context)):
                    context_idx = word_idx[context[k]] if context[k] in word_idx else word_idx[UNK_TAG]
                    y[k, j-i, context_idx] += 1
            residuals = SG.forward(x)
            loss = SG.backward(residuals,x,y)
            epoch_loss += loss
            #print("LOSS")
            #print(loss)

        SG.eval_model()
        print("\n",epoch_loss)
        history.append(epoch_loss)
        np.save(f"weights/skip_w1_{SG.epoch+ep}.npy",SG.w1)
        np.save(f"weights/skip_w2_{SG.epoch+ep}.npy",SG.w2)
    print(history)
