import torch
import numpy as np
import pandas as pd
import torchtext
import math


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


def data(filename="data/tweets.csv.zip"):
    df = pd.read_csv(filename)
    valid = df[df["text"].str.len() >= 50].reset_index()
    return valid["text"]


def sample(probs, temp):
    probs = torch.nn.functional.log_softmax(probs.squeeze(), dim=0)
    probs = (probs / temp).exp()
    probs /= probs.sum()
    probs = probs.cpu().numpy()

    return np.random.choice(np.arange(len(probs)), p=probs)


def generate(model, temp=0.7, start_character=0, end_char=-1):
    model.eval()
    previous_char = start_character
    hidden = None
    with torch.no_grad():
        for _ in range(150):
            inputs = torch.LongTensor([previous_char]).view(1, 1)
            outputs, hidden = model(inputs, hidden)
            sampled = sample(outputs, temp)
            if sampled == end_char:
                return
            yield sampled


class ConvLM(torch.nn.Module):
    def __init__(self, vocab_size, window_size=5, emb_dim=16, filters_count=128):
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

    def forward(self, inputs):
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


class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.text_field = torchtext.data.Field(
            init_token='<s>',
            eos_token='</s>',
            lower=True,
            tokenize=list
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tt = X.apply(self.text_field.preprocess)
        tokenized = tt[tt.str.len() > 50]
        fields = [("text", self.text_field)]
        examples = [
            torchtext.data.Example.fromlist([l], fields)
            for l in tokenized
        ]
        dataset = torchtext.data.Dataset(examples, fields)
        self.text_field.build_vocab(dataset)
        return dataset


def shift(seq, by):
    return torch.cat([seq[by:], seq[:by]])


class MLTrainer(BaseEstimator, TransformerMixin):

    def __init__(self, n_epochs=1):
        self.n_epochs = n_epochs

    def fit(self, X, y=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        vocabulary = X.fields['text'].vocab
        self.model = RnnLM(vocab_size=len(vocabulary))
        name = self.model.__class__.__name__

        batches_count = len(X)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters())

        X_iter, X_iter = torchtext.data.BucketIterator.splits(
            (X, X),
            batch_sizes=(32, 128),
            # shuffle=True,
            device=device,
            sort=False
        )

        for epoch in range(self.n_epochs):
            epoch_loss = 0
            with tqdm(total=batches_count) as progress_bar:
                for i, batch in enumerate(X_iter):
                    logits, _ = self.model(batch.text)
                    targets = shift(batch.text.reshape(-1), by=1)
                    loss = criterion(
                        logits.reshape(-1, logits.shape[-1]), targets)

                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                    optimizer.step()

                    progress_bar.update()
                    progress_bar.set_description(
                        '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                            name, loss.item(),
                            math.exp(loss.item())))

                progress_bar.set_description(
                    '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                        name,
                        epoch_loss / batches_count,
                        math.exp(epoch_loss / batches_count))
                )
        return self


def build_model():
    model = make_pipeline(
        TextTransformer(),
        MLTrainer(),
    )
    return model


def main():
    df = data()
    TextTransformer().fit(df)


if __name__ == '__main__':
    main()
