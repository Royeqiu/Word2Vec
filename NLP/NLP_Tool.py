import spacy
import re
import numpy as np
import math
from spacy.tokens import Doc
class NLP_Tool:

    def __init__(self,load_lg_corpus = True):
        if load_lg_corpus:
            self.nlp = spacy.load('en_core_web_lg')
        else:
            self.nlp = spacy.load('en_core_web_sm')
        self.load_stop_word()

    def load_stop_word(self):
        for word in self.nlp.Defaults.stop_words:
            lex = self.nlp.vocab[word]
            lex.is_stop = True

    def tokenize_sentence(self, sentence):
        return [word.strip() for word in re.split('(\W+)?', sentence) if word.strip()]

    def remove_punctuation(self, text):
        mypunctuation = '!"#$&\'()*+-/:;<=>?@[\\]^_`{|}~*½'
        regex = re.compile('[%s]' % re.escape(mypunctuation))
        text = regex.sub('', text)
        return text

    def get_tokens(self, text):
        doc = self.nlp(text)
        tokens = [token for token in doc if token]
        return tokens

    def get_sents(self, text):
        doc = self.nlp(text)
        return doc.sents

    def get_noun_chunks(self, text):
        doc = self.nlp(text)
        chunks = [chunk for chunk in doc.noun_chunks]
        return chunks

    def get_doc_similarity(self, text1, text2):
        doc = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc.similarity(doc2)

    def get_cos_similarity(self, vec1, vec2):
        vec1_leng=0
        for value in vec1:
            vec1_leng+=(value*value)
        vec1_leng=math.sqrt(vec1_leng)
        vec2_leng=0
        for value in vec2:
            vec2_leng+=(value*value)
        vec2_leng=math.sqrt(vec2_leng)
        product=np.dot(vec1,vec2)

        return product/(vec1_leng*vec2_leng)

    def get_arc_similarity(self,vec1,vec2):
        return (1 - math.acos(self.get_cos_similarity(vec1, vec2)) / math.pi)

    def get_euclidean_distance(self, vec1, vec2):

        return

    def get_phrase_vector(self,text):
        avg_total_vector=np.zeros((300), dtype='f')
        tokens=self.get_tokens(text)
        num=0
        for token in tokens:
            if token.has_vector:
                avg_total_vector+=token.vector
                num+=1
            else:
                continue
        if num!=0:
            return avg_total_vector/num
        else:
            return None

    def is_stop_word(self, word):
        return self.nlp.vocab[word].is_stop

    def transform_pos(self, pos):
        abbr_pos = ''
        if pos == 'VERB':
            abbr_pos = 'v'
        if pos == 'ADJ':
            abbr_pos = 'a'
        if pos == 'ADV':
            abbr_pos = 'adv'
        if pos == 'NOUN':
            abbr_pos = 'n'
        return abbr_pos
    def set_tokenizer(self,Tokenizer,vocab):
        self.nlp.tokenizer = Tokenizer(vocab)

    def get_avg_vector(self,vectors):
        avg_vector=np.zeros((300),dtype='f')
        for vector in vectors:
            avg_vector+=vector
        if len(vectors)==0:
            return avg_vector
        else:
            return avg_vector/len(vectors)



class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

