import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import logging

log = logging.getLogger(__name__)


class TextVectorizer:
    """
    Takes document, applied tokenization and vectorization
    """
    def __init__(self, vocab=15000, length=300):
        self._vectorizer = CountVectorizer(stop_words=None, max_features=vocab)
        self.length = length
        self._analyzer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = self._vectorizer.build_analyzer()

        return self._analyzer

    @property
    def vocabulary(self):
        return self._vectorizer.vocabulary_

    def __call__(self, document):
        try:
            data = []

            for feature in self.analyzer(document):
                featue_idx = self.vocabulary.get(feature)
                if featue_idx is not None:
                    data.append(featue_idx)

            enc1 = np.array(data)
            encoded = np.zeros(self.length, dtype=int)
            length = min(self.length, len(enc1))
            encoded[:length] = enc1[:length]

            return torch.from_numpy(encoded.astype(np.int64))
        except AttributeError:
            raise Exception("Transformer must be fitted!")

    def fit(self, dataset: Dataset):
        documents = [x for x, _ in dataset]
        log.info(f"Fitting Text Vectorizer on {len(documents)} documents")
        self._vectorizer.fit(documents)


class TextPadding:
    """
    Pads each sequence to the same length.

    Taken from Keras.
    """

    def __init__(self, maxlen=1000, dtype='int32', padding='pre', truncating='pre', value=0.0) -> np.ndarray:
        """
        If maxlen is provided, any sequence longer than maxlen is truncated to maxlen.
        Truncation happens off either the beginning(default) or the end of the sequence.
        Supports post - padding and pre - padding(default).

        # Arguments sequences:
        list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

        # Returns
        x: numpy array
        """

        self.maxlen = maxlen
        self.dtype = dtype
        self.padding = padding
        self.truncating = truncating
        self.value = value

    def fit(self, dataset):
        """
        TODO: determine longest sequence
        """
        pass

    def __call__(self, x):
        """

        """
        length = len(x)

        if self.maxlen is None:
            maxlen = np.max(length)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        if len(x) > 0:
            sample_shape = np.asarray(x).shape

        out = (np.ones((self.maxlen) + sample_shape) * self.value).astype(self.dtype)

        if self.truncating == 'pre':
            trunc = x[-maxlen:]
        elif self.truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % self.truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=self.dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence different from expected shape %s' %
                             (trunc.shape[1:], sample_shape))

        if self.padding == 'post':
            out[:len(trunc)] = trunc
        elif self.padding == 'pre':
            out[-len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % self.padding)

        return x