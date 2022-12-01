# adapted from
# https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/dict.py

from collections import defaultdict
import re
import torch

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# simple, fast tokenizer
RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)  

class Dictionary:
    NULL = "__null__"
    UNK = "__unk__"

    @staticmethod
    def re_tokenize(text):
        r"""
        Tokenize using a liberal regular expression.
        Find boundaries between word characters, newlines, and non-word
        non-whitespace tokens ``(r'[\\w\\n]+ | [^\\w\\s] | \\n')``.
        This splits along whitespace and punctuation and keeps the newline as
        a token in the returned list.
        """
        return RETOK.findall(text)

    def filter_stopwords_lemmatize(self, line):
        """
        Uses NLKTs tokenizer and lemmatizer
        Filters nltks stopwords set (not absolute)
        """
        tokens = []
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        if line is not None:

            # split and tokenize
            old_sentence = word_tokenize(line)

            for word in old_sentence:

                # remove punctuation
                # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
                exclude = set(string.punctuation)
                word = ''.join(ch for ch in word if ch not in exclude)

                # if its not empty 
                if word is not None and len(word) > 0:

                    # remove stopwords
                    if word not in stop_words:

                        # lemmatize 
                        # best guess here to treat anything ending in 's'
                        #   as a noun, anything else gets verb treatment
                        new_word = word
                        if word.endswith('s'):
                            new_word = lemmatizer.lemmatize(word)
                        else:
                            new_word = lemmatizer.lemmatize(word, "v")

                        # and add it to the text document
                        tokens.append(new_word)

        return tokens

    def __init__(self, lower=True):
        self.num_docs = 0
        self.freq = defaultdict(int)
        self.doc_freq = defaultdict(int)
        self.tok2ind = {}
        self.ind2tok = {}

        # set up null / unknown tokens
        self.add_token(self.NULL)
        self.add_token(self.UNK)
        self.freq[self.NULL] = 100001
        self.freq[self.UNK] = 100000

        self._unk_idx = self.tok2ind.get(self.UNK)
        self.lower = lower
    
    def __contains__(self, key):
        """
        Return if the dictionary contains the key.
        If key is an int, returns whether the key is in the indices. If key is a str,
        return if the token is in the dict of tokens.
        """
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind
    
    def __getitem__(self, key):
        """
        Lookup the word or ID.
        If key is an int, returns the corresponding token. If it does not exist, return
        the unknown token. If key is a str, return the token's index. If the token is
        not in the dictionary, return the index of the unknown token. If there is no
        unknown token, return ``None``.
        """
        if type(key) == str:
            # return index from token, or unk_token's index, or None
            return self.tok2ind.get(key, self._unk_idx)
        if type(key) == int:
            # return token from index, or unk_token
            return self.ind2tok.get(key, self.UNK)
    
    def __len__(self):
        return len(self.tok2ind)
    
    def __setitem__(self, key, value):
        """
        Set the frequency for a word to a value.
        If the key is not in the dictionary, add it to the dictionary and set its
        frequency to value.
        """
        key = str(key)
        if self.lower:
            key = key.lower()
        self.freq[key] = int(value)
        self.add_token(key)

    def __str__(self):
        """
        Return string representation of frequencies in dictionary.
        """
        return str(self.freq)

    def add_token(self, word):
        """
        Add a single token to the dictionary.
        """
        if word not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[word] = index
            self.ind2tok[index] = word
    
    def add_to_dict(self, tokens):
        """
        Build dictionary from the list of provided tokens.
        """
        self.built = False
        for token in tokens:
            self.add_token(token)
            self.freq[token] += 1
    
    def tokenize(self, text, building=False):
        """
        Return a sequence of tokens from the string.
        """
        if self.lower:
            text = text.lower()

        # word_tokens = self.re_tokenize(text)
        word_tokens = self.filter_stopwords_lemmatize(text)

        return word_tokens
    
    def remove_tail(self, min_freq):
        """
        Remove elements below the frequency cutoff from the dictionary.
        """
        to_remove = []
        for token, freq in self.freq.items():
            if freq < min_freq:
                # queue up removals since can't mutate dict during iteration
                to_remove.append(token)

        for token in to_remove:
            del self.freq[token]
            idx = self.tok2ind.pop(token)
            del self.ind2tok[idx]
    
    def resize_to_max(self, maxtokens):
        """
        Trims the dictionary to the maximum number of tokens.
        """
        if maxtokens >= 0 and len(self.tok2ind) > maxtokens:
            for k in range(maxtokens, len(self.ind2tok)):
                v = self.ind2tok[k]
                del self.ind2tok[k]
                del self.tok2ind[v]
                del self.freq[v]
    
    def sort(self, minfreq=0, maxtokens=0):
        """
        Sort the dictionary.
        Inline operation. Rearranges the dictionary so that the elements with
        the lowest index have the highest counts. This reindexes the dictionary
        according to the sorted frequencies, breaking ties alphabetically by
        token.
        :param bool trim:
            If True, truncate the dictionary based on minfreq and maxtokens.
        """
        # sort first by count, then alphabetically
        if minfreq > 0:
            self.remove_tail(minfreq)
        sorted_pairs = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        new_tok2ind = {}
        new_ind2tok = {}
        for i, (tok, _) in enumerate(sorted_pairs):
            new_tok2ind[tok] = i
            new_ind2tok[i] = tok
        self.tok2ind = new_tok2ind
        self.ind2tok = new_ind2tok
        if maxtokens > 0:
            self.resize_to_max(maxtokens)
        assert len(self.freq) == len(self.ind2tok) == len(self.tok2ind)
        return sorted_pairs

    def txt2vec(self, text: str, vec_type=list):
        """
        Convert a string to a vector (list of ints).
        First runs a sentence tokenizer, then a word tokenizer.
        :param type vec_type:
            The type of the returned vector if the input is a string. Suggested
            ``list``, ``tuple``, ``set``.
        """
        assert isinstance(
            text, str
        ), f'Input to txt2vec must be string, not {type(text)}'

        itr = (self[token] for token in self.tokenize(text))
        if vec_type == list or vec_type == tuple or vec_type == set:
            res = vec_type(itr)
        else:
            raise RuntimeError('Type {} not supported by dict'.format(vec_type))
        return res

    def vec2txt(self, vector, delimiter=' '):
        """
        Convert a vector of IDs to a string.
        Converts a vector (iterable of ints) into a string, with each token separated by
        the delimiter (default ``' '``).
        """
        return delimiter.join(self[int(idx)] for idx in vector)
    
    def ingest_document(self, filename):
        """
        Add all tokens in file to the dictionary and update df metrics (for tfidf).
        """
        self.num_docs += 1
        with open(filename, 'r') as read:
            unique_tokens = set()
            for line in read:
                tokens = self.tokenize(line.strip())
                unique_tokens.update(tokens)
                self.add_to_dict(tokens)
            for token in unique_tokens:
                self.doc_freq[token] += 1
    
    def get_tfidf(self, filename, tf='doc_norm'):
        vec = torch.zeros(len(self))
        with open(filename, 'r') as read:
            for line in read:
                tokens = self.tokenize(line.strip())
                for token in tokens:
                    vec[self[token]] += 1
        
        if tf == 'raw':
            pass  # don't normalize
        elif tf == 'doc_norm':
            vec /= vec.sum()
        else:
            raise RuntimeError(f'Invalid tf argument: {tf}')

        for i in range(len(self)):
            token = self[i]
            if self.doc_freq[token] > 0:
                idf = torch.tensor(self.num_docs / self.doc_freq[token]).log()
            else:
                idf = 0.0
            vec[i] *= idf
        return vec