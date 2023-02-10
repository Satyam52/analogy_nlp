import nltk
from nltk.corpus import stopwords
from skipgram_model import SkipGram
nltk.download('stopwords')
nltk.download('gutenberg')
import numpy as np
from preprocess import preProcess
from nltk.corpus import gutenberg


def getTrainingData(sentences, window_size):
    words = []
    for sentence in sentences:
        for word in sentence:
            if word not in words:
                words.append(word)

    word_dictionary = {}
    i = 0
    for w in words:
        word_dictionary[w] = i
        i += 1
    
    X_train = []
    y_train = []
    V = len(words)

    for sentence in sentences:
        for i, center_word in enumerate(sentence):
            center_word_vector = [0] * V
            center_word_vector[word_dictionary[center_word]] = 1
            context = [0] * V
            for j in range(i - window_size, i + window_size + 1):
                if i != j and j >= 0 and j < len(sentence):
                    context[word_dictionary[sentence[j]]] += 1
            X_train.append(center_word_vector)
            y_train.append(context)
    
    return [X_train, y_train, word_dictionary]

corpus = ""

# with open('sample-text.txt') as f:
#     corpus += f.read()
#     f.close()

#corpus += "The cat sat on the mat, feeling comfortable and relaxed.The cat meowed. The mat was made of woven fibers and provided a soft surface for the cat to rest on. The sun shone brightly in the sky, warming the earth and providing light for all living things. The dog, on the other hand, was active and energetic. It chased the cat, barking loudly as it ran. The cat, startled by the dog's loud barking, quickly ran away. However, the dog was relentless and continued to chase the cat. Despite the dog's efforts, the cat was able to escape and hide. The cat watched from a safe distance as the dog eventually gave up the chase and wandered off. The cat then returned to the mat, grateful for its safety and comfort. The sun continued to shine in the sky, providing warmth and light for all living things. The cat rested on the mat, basking in the sun's warm rays. The dog, meanwhile, explored its surroundings, eager to discover new adventures. As the day progressed, the cat and the dog both found their own ways to enjoy the sun and the warmth of the earth. Whether they were resting or exploring, they both appreciated the gifts that the sun and the earth provided."
epochs = 100
corpus += gutenberg.raw("blake-poems.txt")

training_data = preProcess(corpus)
X_train, y_train, vocab = getTrainingData(training_data,2)

sg = SkipGram(
     X_train = np.array(X_train), 
     y_train = np.array(y_train), 
     word_dictionary = vocab,
     embedding_dimension = 10, 
     learning_rate = 0.001, 
     epochs = epochs
)

sg.train()
np.save('skipgram_weights.npy',sg.W1)
#print(sg.getAllEmbeddings())
# print(sg.predict("cat",2))
# print(sg.predict("elephant",5))
# print(sg.predict("carnivorous",5))
# print(sg.predict("fish",5))
# print(sg.predict("land",5))

# print(sg.analogy("cat","agility","dog"))
# print(sg.analogy("elephant","land","fish"))

print(sg.predict("song",2))