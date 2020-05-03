import nltk
import torch
import numpy as np

np.random.seed(42)


class Tokenizer():

    def __init__(self):
        self.word2ind = None
        self.tag2ind = None

    def fit(self, X, y=None):
        words = {word for sample in X for word, tag in sample}
        self.word2ind = {word: ind + 1 for ind, word in enumerate(words)}
        self.word2ind['<pad>'] = 0

        tags = {tag for sample in X for word, tag in sample}
        self.tag2ind = {tag: ind + 1 for ind, tag in enumerate(tags)}
        self.tag2ind['<pad>'] = 0
        return self

    def transform(self, X, y=None):
        X_ = [[self.word2ind.get(word, 0) for word, _ in sample]
              for sample in X]
        y_ = [[self.tag2ind[tag] for _, tag in sample] for sample in X]
        return X_, y_


def iterate_batches(data, batch_size):
    """
        Return batches in the form [seq_len, batch_size]
    """
    X, y = data
    n_samples = len(X)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_indices = indices[start:end]

        max_sent_len = max(len(X[ind]) for ind in batch_indices)
        X_batch = np.zeros((max_sent_len, len(batch_indices)))
        y_batch = np.zeros((max_sent_len, len(batch_indices)))

        for batch_ind, sample_ind in enumerate(batch_indices):
            X_batch[:len(X[sample_ind]), batch_ind] = X[sample_ind]
            y_batch[:len(y[sample_ind]), batch_ind] = y[sample_ind]

        yield X_batch, y_batch


class LSTMTagger(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, word_emb_dim=100,
                 lstm_hidden_dim=128, lstm_layers_count=1):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, word_emb_dim)
        self._lstm = torch.nn.LSTM(
            word_emb_dim, lstm_hidden_dim, lstm_layers_count)
        self._out_layer = torch.nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, inputs):
        return self._out_layer(self._lstm(self._emb(inputs))[0])


def main():
    nltk.download('brown')
    nltk.download('universal_tagset')

    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    for i, d in enumerate(data):
        print(i, d)
        if i > 5:
            break


if __name__ == '__main__':
    main()
