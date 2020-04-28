import torch
import numpy as np
import pandas as pd
import torchtext
import math


from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Field
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
    def __init__(self, vocab_size, seq_length=128, emb_dim=16, window_size=5):
        super().__init__()

        padding = window_size - 1
        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._conv = torch.nn.Conv2d(1, 1, (window_size, 1),
                                     padding=(padding, 0))

        self._relu = torch.nn.ReLU()
        self._max_pooling = torch.nn.MaxPool2d(
            kernel_size=(window_size + padding - 1, 1))
        self._out_layer = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        '''
        inputs - LongTensor with shape (batch_size, max_word_len)
        outputs - FloatTensor with shape (batch_size,)
        '''
        output = self.embed(inputs.T).max(dim=2)[0].squeeze(dim=1)
        return self._out_layer(output), None

    def embed(self, inputs):

        embs = self._embedding(inputs)
        model = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            self._conv,
            # torch.nn.Dropout(0.2),
            self._relu,
        )
        return model(embs.unsqueeze(dim=1))


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
        return self._out_layer(lstm_out[-1]), hidden


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
        return torchtext.data.Dataset(examples, fields)


def shift(seq, by):
    return torch.cat([seq[by:], seq[:by]])


class MLTrainer(BaseEstimator, TransformerMixin):

    def fit(self, X):
        vocabulary = X.dataset.fields['text'].vocab
        self.model = RnnLM(vocab_size=len(vocabulary))
        name = self.model.__class__.__name__

        batches_count = len(X)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(10):
            epoch_loss = 0
            with tqdm(total=batches_count) as progress_bar:
                for i, batch in enumerate(X):
                    logits, _ = self.model(batch.text)
                    loss = criterion(logits, shift(batch, by=1))

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


def main():
    df = data()
    TextTransformer().fit(df)


if __name__ == '__main__':
    main()
