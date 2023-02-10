import nltk
import wikipediaapi
from nltk.text import Text
from nltk.corpus import wordnet
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen

class Sentences:
    def __init__(self,result,k,word):
        self.result = result
        self.k1 = int(k/2)
        self.k2 = int(k/2)
        self.word = word
        self.multiple_words = []

    def BS_wiki(self):
        url = 'https://en.wikipedia.org/wiki/'+ self.word
        html = urlopen(url) 
        soup = BeautifulSoup(html, 'html.parser')
        op = soup.find(class_="mw-body-content mw-content-ltr")
        div1 = op.find(class_="mw-parser-output")
        ul = div1.find("ul")
        lis = ul.find_all("li")
        for li in lis:
            if len(self.multiple_words)<3:
                a= li.find('a')
                title = a.get('title')
                self.multiple_words.append(title)


    # This function gets sentences from wikipedia pages based on the chosen word
    def get_wiki_sentences(self):
        wiki_wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        p_wiki = wiki_wiki.page(self.word)
        if "disambiguation" in p_wiki.text:
            self.BS_wiki()
            sentences = []
            for w in self.multiple_words:
                vals = wiki_wiki.page(w).text.split(".")
                vals = [str(v)for v in vals if re.search(self.word,str(v),re.IGNORECASE)]
                self.result += vals[:int(self.k2/3)]
        else:
            sentences = p_wiki.text.split(".")
            sentences = [str(sentence) for sentence in sentences if re.search(self.word,str(sentence),re.IGNORECASE)]
            self.result += sentences[:self.k2]

        self.result = list(set(self.result))

    # The following functions get sentences based on Synsets, Hypernyms, Hyponyms, Antonyms, etc
    def get_sentences_from_syns(self,pool):
        for syn in pool:
            definition = syn.definition()
            sentences = definition.split("\n")
            sentences = [sentence for sentence in sentences if re.search(self.word,sentence,re.IGNORECASE)]
            if len(sentences):
                self.wn_result += sentences
            examples = syn.examples()
            sentences = [example for example in examples if re.search(self.word,example,re.IGNORECASE)]
            if len(sentences):
                self.wn_result += sentences

    def get_sentences_from_lemmas(self,lemma):
        synonyms = wordnet.synsets(lemma)
        for syn in synonyms:
            definition = syn.definition()
            sentences = definition.split("\n")
            sentences = [sentence for sentence in sentences if re.search(self.word,sentence,re.IGNORECASE)]
            if len(sentences):
                self.wn_result += sentences
            examples = syn.examples()
            sentences = [example for example in examples if re.search(self.word,example,re.IGNORECASE)]
            if len(sentences):
                self.wn_result += sentences

            h_syn = syn.hypernyms()
            self.get_sentences_from_syns(h_syn)

            h_syn = syn.hyponyms()
            self.get_sentences_from_syns(h_syn)

            h_syn = syn.member_holonyms()
            self.get_sentences_from_syns(h_syn)

            h_syn = syn.root_hypernyms()
            self.get_sentences_from_syns(h_syn)

    def get_wordnet_sentences(self):
        synonyms = wordnet.synsets(self.word)
        self.wn_result = []

        for syn in synonyms:
            definition = syn.definition()
            sentences = definition.split("\n")
            sentences = [sentence for sentence in sentences if re.search(self.word,sentence,re.IGNORECASE)]
            if len(sentences):
                self.wn_result += sentences
            examples = syn.examples()
            sentences = [example for example in examples if re.search(self.word,example,re.IGNORECASE)]
            if len(sentences):
                self.wn_result += sentences

            h_syn = syn.hypernyms()
            self.get_sentences_from_syns(h_syn)

            h_syn = syn.hyponyms()
            self.get_sentences_from_syns(h_syn)

            h_syn = syn.member_holonyms()
            self.get_sentences_from_syns(h_syn)

            h_syn = syn.root_hypernyms()
            self.get_sentences_from_syns(h_syn)

            for l in syn.lemmas():
            
                for a in l.antonyms():
                    self.get_sentences_from_lemmas(a.name())

                for a in l.pertainyms():
                    self.get_sentences_from_lemmas(a.name())

                for a in l.derivationally_related_forms():
                    self.get_sentences_from_lemmas(a.name())

        self.result += self.wn_result[:self.k1]
        if(len(self.wn_result)<self.k1):
            self.k2 += self.k1 - len(self.wn_result)
        


