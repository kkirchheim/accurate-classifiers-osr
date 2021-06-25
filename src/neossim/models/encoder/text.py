from torch import nn
from neossim.models.encoder.img.encoder import EncoderBase


class BOWEmbedding(EncoderBase):
    """
    Bag of Word Embedding
    """

    def __init__(self, vocab, n_features):
        super(BOWEmbedding, self).__init__(n_features=n_features)
        self.embedding = nn.EmbeddingBag(vocab, embedding_dim=n_features)

    def forward(self, x):
        return self.embedding(x)


class WordEmbedding(EncoderBase):
    """
    Word Embedding
    """
    def __init__(self, vocab, n_features):
        super(WordEmbedding, self).__init__(n_features=n_features)
        self.embedding = nn.Embedding(vocab, embedding_dim=n_features)

    def forward(self, x):
        return self.embedding(x)
