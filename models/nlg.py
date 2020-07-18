import math
import torch
import skorch
import numpy as np
import pandas as pd


from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from torchtext.data import Dataset, Example, Field
from torchtext.data import BucketIterator


"""
# Setup
!curl -k -L "https://drive.google.com/uc?export=download&id=1Pq4aklVdj-sOnQw68e1ZZ_ImMiC8IR1V" -o data/tweets.csv.zip
!pip install -r numpy torch pandas sklearn tqdm
"""  # noqa

"""
Problems:

- [x] Fixed window CNN based language model
- [x] Text generation from the language model
- [x] Implement language model targets and loss function
- [x] Mask the unknown words and pad them
- [x] Recurrent language model
- [x] Add sampling from the recurrent model
- [ ] Try different optimizer:
        optim.SGD(model.parameters(), lr=20., weight_decay=1e-6)
- [x] Add inverse transform
- [x] Variational (Locked) dropout
- [ ] Conditional text generation
- [ ] Try another dataset
"""


def data(filename="data/tweets.csv.zip"):
    df = pd.read_csv(filename)
    valid = df[df["text"].str.len() >= 50].reset_index()
    return valid["text"]


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=1):
        self.fields = fields
        self.min_freq = min_freq

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        return Dataset(examples, self.fields)

    def inverse_transform(self, X):
        strings = np.take(self.fields[0][-1].vocab.itos, X)
        output = []
        for seq in strings:
            output.append("".join(seq))
        return output


def sample(probs, temp):
    probs = torch.nn.functional.log_softmax(probs.squeeze(), dim=0)
    probs = (probs / temp).exp()
    probs /= probs.sum()
    probs = probs.cpu().numpy()

    return np.random.choice(np.arange(len(probs)), p=probs)


def generate(model, temp=0.8, size=150, prev_token=1, end_token=2):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        hidden = None
        for _ in range(size):
            inputs = torch.LongTensor([[prev_token]]).to(device)
            probs, hidden = model(inputs, hidden)
            prev_token = sample(probs, temp)
            if prev_token == end_token:
                return prev_token
            yield prev_token


class ConvLM(torch.nn.Module):
    def __init__(self, vocab_size, window_size=5,
                 emb_dim=16, filters_count=128):
        super().__init__()

        self._window_size = window_size

        self._embs = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=1)
        self._conv = torch.nn.Sequential(
            torch.nn.Conv1d(emb_dim, filters_count,
                            kernel_size=self._window_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(filters_count)
        )
        self._output = torch.nn.Linear(filters_count, vocab_size)

    def forward(self, inputs, hidden=None):
        # Left-side paddings
        padding = inputs.new_zeros((self._window_size - 1, inputs.shape[1]))

        # Cat for correct convolutions
        inputs = torch.cat((padding, inputs), 0)

        # Embeddings
        embs = self._embs(inputs)

        # Permute for further convolutions
        embs = embs.permute((1, 2, 0))

        # Convolutions
        output = self._conv(embs)

        # Permute for output layer
        output = output.permute((2, 0, 1))

        return self._output(output), None


class RnnLM(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=16, lstm_hidden_dim=128):
        super().__init__()

        self.vocab_size = vocab_size
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.LSTM(
            input_size=emb_dim, hidden_size=lstm_hidden_dim)
        self._out_layer = torch.nn.Linear(lstm_hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self._emb(inputs)
        lstm_out, hidden = self._rnn(embedded, hidden)
        return self._out_layer(lstm_out), hidden


class LockedDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, dropout=0.5):
        if not self.training or not dropout:
            return inputs

        _, batch_size, in_dim = inputs.shape
        masks = torch.bernoulli(dropout * torch.ones((1, batch_size, in_dim)))
        masks *= 1. / (.1 - dropout)
        return inputs * masks


def shift(seq, by):
    return torch.cat([seq[by:], seq.new_ones((by, seq.shape[1]))])

class LanguageModelNet(skorch.NeuralNet):

    def get_loss(self, y_pred, y_true, X=None, training=False):
        out, _ = y_pred
        logits = out.view(-1, out.shape[-1])
        return self.criterion_(logits, shift(y_true.T, by=1).view(-1))

class MLTrainer(BaseEstimator, TransformerMixin):

    def __init__(self, mtype=None, n_epochs=5, batch_size=32):
        self.n_epochs = n_epochs
        self.mtype = mtype or RnnLM
        self.batch_size = batch_size

    def fit(self, X, y=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        vocabulary = X.fields['text'].vocab
        self.model = self.mtype(vocab_size=len(vocabulary)).to(device)
        name = self.model.__class__.__name__

        criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device)
        optimizer = torch.optim.Adam(self.model.parameters())

        X_iter = BucketIterator(
            X,
            batch_size=self.batch_size,
            device=device,
            sort=False
        )

        pad_token = X_iter.dataset.fields["text"].vocab.stoi['<pad>']
        unk_token = X_iter.dataset.fields["text"].vocab.stoi['<unk>']

        for epoch in range(self.n_epochs):
            epoch_loss, batch_size = 0, len(X_iter)
            t = tqdm(X_iter, total=batch_size)
            for i, batch in enumerate(t):
                logits, _ = self.model(batch.text)
                # import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa
                targets = shift(batch.text, by=1).view(-1)
                loss_vectors = criterion(
                    logits.reshape(-1, logits.shape[-1]), targets)

                idx = ((targets != pad_token) & (targets != unk_token))
                # Average: sum of all divided by number of unmasked
                loss = (loss_vectors * idx).sum() / (
                    targets.shape[0] - (~idx).sum() + 1
                )

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()

                t.set_description(
                    '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                        name, loss.item(),
                        math.exp(loss.item())))

            t.set_description(
                '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                    name, epoch_loss / batch_size,
                    math.exp(epoch_loss / batch_size)))
        return self

    def inverse_transform(self, X):
        output = []
        for temp, seqsize, tstart, tend in X:
            symbols = list(generate(self.model, temp, seqsize, tstart, tend))
            output.append(np.squeeze(symbols))
        return np.array(output)


def build_preprocessor():
    text_field = Field(
        batch_first=True,
        init_token='<s>',
        eos_token='</s>',
        lower=True,
        tokenize=list,
    )

    fields = [
        ("text", text_field),
    ]
    return TextPreprocessor(fields)


class LanguageModelIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.text, batch.text


def build_model(module=RnnLM):
    model = LanguageModelNet(
        module=module,
        module__vocab_size=100,  # Dummy dimension
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=32,
        iterator_train=LanguageModelIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=LanguageModelIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            skorch.callbacks.GradientNormClipping(1.),
            # DynamicParSertter(),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    df = data()

    cnn_model = build_model(mtype=ConvLM, n_epochs=30)
    cnn_model.fit(df, None)

    tstart = cnn_model[0].text_field.vocab["<s>"]
    tend = cnn_model[0].text_field.vocab["</s>"]

    cnn_generated = cnn_model.inverse_transform([[0.7, 100, tstart, tend]])
    print(cnn_generated)

    rnn_model = build_model(n_epochs=30)
    rnn_model.fit(df, None)
    rnn_generated = rnn_model.inverse_transform([[0.7, 100, tstart, tend]])
    print(rnn_generated)


if __name__ == '__main__':
    main()
