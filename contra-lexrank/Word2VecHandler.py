import gensim
import numpy as np
from nltk.tokenize import word_tokenize

from shared.Argument import Argument


@DeprecationWarning
class Word2VecHandler:
    """
    Handles the conversion from words, sentences, and documents to numeric embeddings.
    """

    def __init__(self, w2v_path):
        """
        Initializes the object in terms of loading the pretrained embeddings.
        Maybe you need to download the model first:
        [download](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
        :param w2v_path: path to the model
        """
        self.w2v_path = w2v_path
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            self.w2v_path, binary=True)

    def word_vector(self, word: str):
        """
        Gets the word vector from the underlying model.
        :param word: word for which the vector is wanted
        :return: the word embedding
        """
        if word in self.model.vocab:
            return self.model[word]
        else:
            # Todo: Maybe similar word
            return None

    def sentence_vector(self, sentence, tokenize=False):
        """
        Gets the vector for a sentence by averaging all word vectors.
        :param tokenize: indicates whether tokenization should be applied
        :param sentence: either list of tokens or string that needs to be tokenized
        :return: the sentence embedding
        """
        if tokenize:
            sentence = self.tokenize(sentence)

        wv = np.array(list(filter(lambda v: v is not None, map(self.word_vector, sentence))))
        return np.mean(wv, axis=0)

    def document_vector(self, argument: Argument, tokenize=False):
        """
        Gets the vector for a document by averaging all word vectors.
        :param argument: either a list of sentences that needs to be tokenized or a list of lists of tokens
        :param tokenize: indicates whether tokenization should be applied
        :return: the document embedding
        """
        # Check whether the sentence embeddings already exist.
        if len(argument.sentence_embeddings) == 0:
            _sent_emb = np.array([self.sentence_vector(s, tokenize) for s in argument.sentences])
        else:
            _sent_emb = np.array(argument.sentence_embeddings)

        # Aggregation to argument/document embeddings.
        return np.mean(_sent_emb, axis=0)

    @staticmethod
    def tokenize(text: str) -> list:
        """
        Tokenizes the given text.
        :param text: string to be tokenized
        :return: list of tokens
        """
        return word_tokenize(text)
