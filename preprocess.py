import string
import nltk
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


nltk.download('punkt')



def preProcess(corpus):
    corpus = corpus.lower()
    corpus = re.sub(r'\n\t\s\d+', '', corpus)
    sentences = corpus.split(".")
    translator = str.maketrans('', '', string.punctuation)
    sentences = [" ".join(sentence.translate(translator) .split()) for sentence in sentences]
    sentences = [word_tokenize(remove_stopwords(sentence)) for sentence in sentences]
    stop_words = set(stopwords.words("english"))
    for i in range(0,len(sentences)):
        sentences[i] = [word for word in sentences[i] if word not in stop_words]
    return sentences



