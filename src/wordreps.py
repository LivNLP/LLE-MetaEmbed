
"""
This module implements the WordReps class that can load pre-trained word embeddings
in binary or text format.

Danushka Bollegala
12 August 2016.
"""

import numpy


class WordReps:

    def __init__(self):
        self.vocab = None 
        self.vects = None 
        self.vector_size = None
        pass


    def read_model(self, fname, dim, words=None, HEADER=False, case_sensitive=False):
        """
        Read the word vectors where the first token is the word.
        """
        res = {}
        F = open(fname)
        if HEADER:
            res["method"] = F.readline().split('=')[1].strip()
            res["input"] = F.readline().split('=')[1].strip()
            res["rank"] = int(F.readline().split('=')[1])
            res["itermax"] = int(F.readline().split('=')[1])
            res["vertices"] = int(F.readline().split('=')[1])
            res["edges"] = int(F.readline().split('=')[1])
            res["labels"] = int(F.readline().split('=')[1])
            R = res["rank"]
        R = dim
        # read the vectors.
        vects = {}
        vocab = []
        line = F.readline()

        # Check whether the first line contains the number of words and the dimensionality.
        # If so, skip it.
        if len(line.split()) == 2:
            line = F.readline()
            
        while len(line) != 0:
            p = line.split()
            word = p[0]
            if not case_sensitive:
                word = word.lower()
            if (words is None) or (word in words):
                v = numpy.zeros(R, float)
                for i in range(0, R):
                    v[i] = float(p[i+1])
                vects[word] = vects.get(word, numpy.zeros(R, float)) + v
                vocab.append(word)
            line = F.readline()
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.dim = R
        pass


    def read_w2v_model_text(self, fname, dim):
        """
        Read the word vectors where the first token is the word.
        """
        F = open(fname)
        R = dim
        # read the vectors.
        vects = {}
        vocab = []
        line = F.readline()  # vocab size and dimensionality 
        assert(int(line.split()[1]) == R)
        line = F.readline()
        while len(line) != 0:
            p = line.split()
            word = p[0]
            v = numpy.zeros(R, float)
            for i in range(0, R):
                v[i] = float(p[i+1])
            vects[word] = normalize(v)
            vocab.append(word)
            line = F.readline()
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.dim = R
        pass


    def read_w2v_model_binary(self, fname, dim):
        """
        Given a model file (fname) produced by word2vect, read the vocabulary list 
        and the vectors for each word. We will return a dictionary of the form
        h[word] = numpy.array of dimensionality.
        """
        F = open(fname, 'rb')
        header = F.readline()
        vocab_size, vector_size = map(int, header.split())
        vocab = []
        vects = {}
        print "Vocabulary size =", vocab_size
        print "Vector size =", vector_size
        assert(dim == vector_size)
        binary_len = numpy.dtype(numpy.float32).itemsize * vector_size
        for line_number in xrange(vocab_size):
            # mixed text and binary: read text first, then binary
            word = ''
            while True:
                ch = F.read(1)
                if ch == ' ':
                    break
                if ch != '\n':
                    word += ch
            word = word.lower()
            vocab.append(word)
            vector = numpy.fromstring(F.read(binary_len), numpy.float32)
            vects[word] = vector        
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.dim = vector_size
        pass


    def get_vect(self, word):
        if word not in self.vocab:
            return numpy.zeros(self.vector_size, float)
        return self.vects[word]


    def normalize_all(self, w=1.0):
        """
        L2 normalizes all vectors. If the weight w is specified we will multiple
        the L2 normalized vectors by this weight. This is useful for emphasizing
        some source embeddings when meta-embedding.
        """
        for word in self.vocab:
            self.vects[word] = w * normalize(self.vects[word])
        pass


    def test_model(self):
        A = self.get_vect("man")
        B = self.get_vect("king")
        C = self.get_vect("woman")
        D = self.get_vect("queen")
        x = B - A + C
        print cosine(x, D)
        pass   


def cosine(x, y):
    """
    Compute the cosine similarity between two vectors x and y. 
    We must L2 normalize x and y before we use this function.
    """
    #return numpy.dot(x,y.T) / (numpy.linalg.norm(x) * numpy.linalg.norm(y))
    norm = numpy.linalg.norm(x) * numpy.linalg.norm(y)
    return 0 if norm == 0 else (numpy.dot(x, y) / norm)


def normalize(x):
    """
    L2 normalize vector x. 
    """
    norm_x = numpy.linalg.norm(x)
    return x if norm_x == 0 else (x / norm_x)


def get_embedding(word, WR):
    """
    If we can find the embedding for the word in vects, we will return it.
    Otherwise, we will check if the lowercase version of word appears in vects
    and if so we will return the embedding for the lowercase version of the word.
    Otherwise we will return a zero vector.
    """
    if word in WR.vects:
        return WR.vects[word]
    else:
        return numpy.zeros(WR.dim, dtype=float)

