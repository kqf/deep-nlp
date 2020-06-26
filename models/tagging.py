import nltk
import torch
import skorch
import random
import numpy as np
import pandas as pd
import gensim.downloader as gapi

from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Field, Example, Dataset, BucketIterator

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


"""
Takeaways:

- [x] Sequence tagging can be achieved by making use of LSTM output
- [x] Input [seq_size, batch_size] -> [batch_size, seq_size, n_targets]
- [x] It is possible to mask the padding tokens:
    - Use `ignore_index` in criterion
    - Calculate `mask = y_batch != pad_index` to evaluate the model
- [x] Masking should give more accurate evaluation (accuracy 95.3%)
- [x] The bidirectional LSTM improves the result (accuracy 96.8%)
- [x] Pretrained embeddings (accuracy 96%)
- [x] Unfreeze the pretrained embeddings (accuracy 96%):
    - Beware to use the same embeddings on train and test set
    - use loss += torch.dist(trainable, original)
- [x] It is possible to make char embeddings (accuracy 95%)
- [ ] TODO: BERT, no tortext?

"""


def to_pandas(raw):
    return pd.DataFrame((zip(*r) for r in raw), columns=["tokens", "tags"])


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=0):
        self.fields = fields
        self.min_freq = min_freq or {}

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        dataset = Dataset(examples, self.fields)
        return dataset


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

    @property
    def emb_size(self):
        return len(self.word2ind)


class EmbeddingsTokenizer(Tokenizer):

    def __init__(self, w2v):
        self.w2v = w2v
        self.embeddings = None

    def fit(self, X, y=None):
        super().fit(X, y)

        self.embeddings = np.zeros(
            (len(self.word2ind), self.w2v.vectors.shape[1]))

        for word, ind in self.word2ind.items():
            word = word.lower()
            if word in self.w2v.vocab:
                self.embeddings[ind] = self.w2v.get_vector(word)
        return self

    @property
    def emb_size(self):
        return self.embeddings


class LSTMTagger(torch.nn.Module):
    def __init__(self, vocab_size=1, tagset_size=1, word_emb_dim=100,
                 lstm_hidden_dim=128, lstm_layers_count=1):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, word_emb_dim)
        self._lstm = torch.nn.LSTM(
            word_emb_dim, lstm_hidden_dim, lstm_layers_count)
        self._out_layer = torch.nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, inputs):
        return self._out_layer(self._lstm(self._emb(inputs))[0])


class PretrainedEmbLSTMTagger(torch.nn.Module):
    def __init__(self, embeddings, tagset_size, word_emb_dim=100,
                 lstm_hidden_dim=128, lstm_layers_count=1):
        super().__init__()

        self._emb = torch.nn.Embedding.from_pretrained(embeddings)
        self._lstm = torch.nn.LSTM(
            word_emb_dim, lstm_hidden_dim, lstm_layers_count)
        self._out_layer = torch.nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, inputs):
        return self._out_layer(self._lstm(self._emb(inputs))[0])


class BiLSTMTagger(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, word_emb_dim=100,
                 lstm_hidden_dim=128, lstm_layers_count=1):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, word_emb_dim)
        self._lstm = torch.nn.LSTM(
            word_emb_dim, lstm_hidden_dim, lstm_layers_count,
            bidirectional=True)
        self._out_layer = torch.nn.Linear(lstm_hidden_dim * 2, tagset_size)

    def forward(self, inputs):
        return self._out_layer(self._lstm(self._emb(inputs))[0])


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        svocab = X.fields["tokens"].vocab
        vocab = X.fields["tags"].vocab

        net.set_params(module__vocab_size=len(svocab))
        net.set_params(module__tagset_size=len(vocab))
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_preprocessor():
    fields = [
        ("tokens", Field()),
        ("tags", Field(is_target=True))
    ]
    return TextPreprocessor(fields, 1)


class VectorField(Field):
    def __init__(self, emb_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_file = emb_file

    def build_vocab(self, *args, **kwargs):
        return super().build_vocab(*args, vectors=self.emb_file, **kwargs)


def build_preprocessor_emb(emb_file="glove.6B.50d"):
    fields = [
        ("tokens", VectorField(emb_file)),
        ("tags", Field(is_target=True))
    ]
    return TextPreprocessor(fields, 1)


class TaggerNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred.view(-1, y_pred.shape[-1])
        return self.criterion_(logits, y_true.view(-1))


def build_model():
    model = TaggerNet(
        module=LSTMTagger,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=32,
        iterator_train=BucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=BucketIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            DynamicVariablesSetter(),
            skorch.callbacks.GradientNormClipping(1.),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def build_embedding_model():
    tokenizer = EmbeddingsTokenizer(w2v=gapi.load('glove-wiki-gigaword-100'))
    return make_pipeline(tokenizer, PretrainedEmbLSTMTagger(tokenizer))


def main():
    nltk.download('brown')
    nltk.download('universal_tagset')

    data = to_pandas(nltk.corpus.brown.tagged_sents(tagset='universal'))
    print(data.head())

    model = build_model()
    model.fit(data)

    emodel = build_embedding_model()
    emodel.fit(data)


if __name__ == '__main__':
    main()
