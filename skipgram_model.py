import numpy as np
from nltk.stem.porter import PorterStemmer



def glorot_initialization(size):
    n1 = size[0]
    n2 = size[1]
    #w = np.full((n1, n2),0)
    standard_dev = np.sqrt(2.0 / (n1 + n2))
    w = np.random.normal(0,standard_dev, size=(n1, n2))
    return w

def cosine_similarities(vector, matrix):
    return np.dot(matrix, vector) / np.linalg.norm(matrix, axis=1) / np.linalg.norm(vector)

# (dog,[bark,loyal]) will correspond to the training instance 
# 0 - dog, 1 - bark, 2 - loyal
# [1,0,0] , [0,1,1]
class SkipGram:
    def __init__(self, X_train, y_train, word_dictionary,embedding_dimension, learning_rate, epochs):
        self.V = len(X_train[0])
        self.N = len(X_train)
        self.D = embedding_dimension
        self.X_train = X_train
        self.y_train = y_train
        self.W1 = glorot_initialization([self.V,self.D])
        self.W2 = glorot_initialization([self.D,self.V])
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.word_dictionary = word_dictionary

    def softmax(self,X):
        exp_X = np.exp(X)
        sum_exp_X = np.sum(exp_X)
        softmax_X = exp_X/sum_exp_X
        return softmax_X

    def forward(self,X):
        self.l1 = np.dot(self.W1.T,X)
        self.l2 = np.dot(self.W2.T,self.l1)
        self.y = self.softmax(self.l2)
        return self.y

    def backward(self,X,T):
        error = self.y - T
        self.l1 = self.l1.reshape(self.D,1)
        error = error.reshape(self.V,1)
        dW2 = np.dot(self.l1,error.T)
        X = X.reshape(self.V,1)
        dW1 = np.dot(X,np.dot(self.W2,error).T)
        self.W1 -= self.learning_rate*dW1
        self.W2 -= self.learning_rate*dW2
    
    def train(self):
        for e in range(1,self.epochs):
            self.epoch_loss = 0
            for i in range(self.N):
                self.forward(self.X_train[i])
                self.backward(self.X_train[i], self.y_train[i])
                C = sum([1 for k in range(self.V) if self.y_train[i][k]])
                self.epoch_loss = self.epoch_loss - sum([self.l2[k] for k in range(self.V) if self.y_train[i][k]]) 
                self.epoch_loss = self.epoch_loss + C * np.log(np.sum(np.exp(self.l2)))
            print("Epoch ",e,"      Loss:",self.epoch_loss)

    def getAllEmbeddings(self):
        return self.W1

    def getWordEmbedding(self,word):
        if word not in self.word_dictionary.keys():
            print("Word not present in dictionary")
            return None
        else:
            ind = self.word_dictionary[word]
            return self.W1[ind]

    def analogy(self,word1, word2, given):
        stemmer = PorterStemmer()
        word1_vec = self.getWordEmbedding(stemmer.stem(word1))
        word2_vec = self.getWordEmbedding(stemmer.stem(word2))
        given_vec = self.getWordEmbedding(stemmer.stem(given))
        target_vec = word2_vec - word1_vec + given_vec
        similarity_measure = cosine_similarities(target_vec, self.W1)
        word = [k for k, v in self.word_dictionary.items() if v == np.argmax(similarity_measure)][0]
        return word

    def predict(self, word, count):
        stemmer = PorterStemmer()
        word = stemmer.stem(word)
        if word not in self.word_dictionary:
            print("Word not found in dictionary")
            return None
        
        index = self.word_dictionary[word]
        X = [0] * self.V
        X[index] = 1
        prediction = self.forward(X)
        
        sorted_output = sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)
        words = [list(self.word_dictionary.keys())[i[0]] for i in sorted_output[:count]]
        return words




            


