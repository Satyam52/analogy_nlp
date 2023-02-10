from extract import Sentences
import string


# This function gets a set of unique words from the Analogy_dataset.txt file
def get_unique_words(filename):
    with open(filename, "r") as file:
        words = file.read().split()
    words = [word.translate(str.maketrans("", "", string.punctuation)).lower() for word in words]
    return list(set(words))

unique_words = get_unique_words("../Analogy_dataset.txt")
sentences_list = []
K = 30
for word in unique_words:
    obj = Sentences([],K,word)
    obj.get_wordnet_sentences()
    obj.get_wiki_sentences()
    sentences_list += obj.result

file = open('../analogy_corpus.txt','w')
for obj in sentences_list:
    file.write(obj+"\n")
file.close()


