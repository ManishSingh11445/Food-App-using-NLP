import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np
#nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word(tokenized_sentence,all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag



