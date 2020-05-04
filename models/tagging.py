import math
import nltk
import torch
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import make_pipeline

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

    @property
    def padding(self):
        return self.word2ind["<pad>"]


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


def epoch(model, criterion, data, batch_size, optimizer=None,
          name=None, pi=None):
    epoch_loss = 0
    correct_count = 0
    sum_count = 0

    is_train = optimizer is not None
    name = name or ''
    model.train(is_train)

    batches_count = math.ceil(len(data[0]) / batch_size)

    with torch.autograd.set_grad_enabled(is_train):
        bar = tqdm(iterate_batches(data, batch_size), total=batches_count)
        for i, (X_batch, y_batch) in enumerate(bar):
            X_batch = torch.LongTensor(X_batch)
            y_batch = torch.LongTensor(y_batch)
            logits = model(X_batch)

            loss = criterion(
                logits.view(-1, logits.shape[-1]), y_batch.view(-1))

            epoch_loss += loss.item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mask = (y_batch != pi) if pi is not None else torch.ones_like(y_batch)

            preds = torch.argmax(logits, dim=-1)
            cur_correct_count = ((preds == y_batch) * mask).sum().item()
            cur_sum_count = mask.sum().item()

            correct_count += cur_correct_count
            sum_count += cur_sum_count

            bar.update()
            bar.set_description(
                '{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
                    name, loss.item(), cur_correct_count / cur_sum_count)
            )

        bar.set_description(
            '{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
                name, epoch_loss / batches_count, correct_count / sum_count)
        )

    return epoch_loss / batches_count, correct_count / sum_count


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


class TaggerModel():
    def __init__(self, tokenizer, batch_size=64, epochs_count=20):
        self.epochs_count = epochs_count
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMTagger(
            len(self.tokenizer.word2ind),
            len(self.tokenizer.tag2ind)).to(device)

        pi = self.tokenizer.padding
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pi).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        for i in range(self.epochs_count):
            name_prefix = '[{} / {}] '.format(i + 1, self.epochs_count)
            epoch(
                self.model,
                self.criterion,
                X,
                self.batch_size,
                self.optimizer,
                name_prefix,
                pi=pi, # padding index
            )

        return self


def build_model():
    tokenizer = Tokenizer()
    return make_pipeline(tokenizer, TaggerModel(tokenizer))


def main():
    nltk.download('brown')
    nltk.download('universal_tagset')

    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    model = build_model()
    model.fit(data)


if __name__ == '__main__':
    main()
