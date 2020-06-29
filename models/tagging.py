import time
import nltk
import torch
import skorch
import random
import itertools
import numpy as np
import pandas as pd
from functools import partial
from contextlib import contextmanager

from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from torchtext.data import Field, Example, Dataset, BucketIterator
from transformers import DistilBertTokenizer, DistilBertModel

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


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{color}[{name}] done in {et:.0f} s{nocolor}".format(
        name=name, et=time.time() - t0,
        color='\033[1;33m', nocolor='\033[0m'))


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
        return Dataset(examples, self.fields)


class LSTMTagger(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_dim=100,
                 lstm_hidden_dim=128,
                 lstm_layers_count=1, bidirectional=False):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._lstm = torch.nn.LSTM(
            emb_dim, lstm_hidden_dim,
            lstm_layers_count, bidirectional=bidirectional)

        hidden = 2 * lstm_hidden_dim if bidirectional else lstm_hidden_dim
        self._out_layer = torch.nn.Linear(hidden, tagset_size)

    def forward(self, inputs):
        return self._out_layer(self._lstm(self._emb(inputs))[0])


class PretrainedEmbLSTMTagger(torch.nn.Module):
    def __init__(self, emb, tagset_size, emb_dim=100,
                 lstm_hidden_dim=128,
                 lstm_layers_count=1, bidirectional=False):
        super().__init__()
        if emb is not None:
            self._emb = torch.nn.Embedding.from_pretrained(emb)

        self._lstm = torch.nn.LSTM(
            emb_dim, lstm_hidden_dim,
            lstm_layers_count, bidirectional=bidirectional)

        hidden = 2 * lstm_hidden_dim if bidirectional else lstm_hidden_dim
        self._out_layer = torch.nn.Linear(hidden, tagset_size)

    def forward(self, inputs):
        return self._out_layer(self._lstm(self._emb(inputs))[0])


class DynamicTagSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        self.setup_embeddings(net, X.fields["tokens"])

        vocab = X.fields["tags"].vocab
        net.set_params(module__tagset_size=len(vocab))
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    def setup_embeddings(self, net, vocab):
        pass

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DynamicVocabSetter(DynamicTagSetter):
    def setup_embeddings(self, net, field):
        return net.set_params(module__vocab_size=len(field.vocab))


class DynamicEmbSetter(DynamicTagSetter):
    def setup_embeddings(self, net, field):
        net.set_params(module__emb=field.vocab.vectors)
        return net.set_params(module__emb_dim=field.vocab.vectors.shape[-1])


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
        return super().build_vocab(
            *args,
            vectors=self.emb_file,
            unk_init=torch.Tensor.normal_,
            **kwargs
        )


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

    def _predict(self, X):
        idx = self.predict_proba(X).argmax(-1)
        # NB: torchtext batches the sentences together
        return idx.reshape(len(X), -1)

    def predict(self, X):
        idx = self._predict(X)
        return np.take(X.fields["tags"].vocab.itos, idx)

    def score(self, X, y):
        idx = self._predict(X)

        # Ensure seq length
        y_pred = itertools.chain(*[yp[:len(yt)] for yp, yt in zip(idx, y)])

        # Convert answers to index (according to the new sklearn interface)
        stoi = X.fields["tags"].vocab.stoi
        y_true = [stoi[t] for yt in y for t in yt]
        return f1_score(y_true, list(y_pred), average="micro")


def build_model():
    model = TaggerNet(
        module=LSTMTagger,
        module__vocab_size=1,  # Dummy dimension
        module__tagset_size=1,
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
            DynamicVocabSetter(),
            skorch.callbacks.GradientNormClipping(1.),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def build_emb_model():
    model = TaggerNet(
        module=PretrainedEmbLSTMTagger,
        module__emb=None,
        module__tagset_size=1,
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
            DynamicEmbSetter(),
            skorch.callbacks.GradientNormClipping(1.),
        ],
    )

    full = make_pipeline(
        build_preprocessor_emb(),
        model,
    )
    return full


class BERTTagger(torch.nn.Module):
    def __init__(self, bert, tagset_size, dropout=0.25):
        super().__init__()
        self._bert = bert
        emb_dim = bert.config.to_dict()['dim']
        self._out_layer = torch.nn.Linear(emb_dim, tagset_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs -> [batch_size, seq_len]
        inputs = inputs.permute(1, 0)

        # embedded[batch_size, seq_len, emb_dim]
        embedded = self.dropout(self._bert(inputs)[0])

        # embedded -> [seq_len, batch_size, emb_dim]
        embedded = embedded.permute(1, 0, 2)

        # predictions = [seq_len, batch_size, tagset_size]
        # import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa
        return self._out_layer(self.dropout(embedded))


def bert_text_preprocess(tokens, tokenizer, max_len):
    tokens = tokens[:max_len - 1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens


def bert_tag_preprocessor(tokens, max_len):
    tokens = tokens[:max_len - 1]
    return tokens


def build_bert_preprocessor(modelname='distilbert-base-cased', max_len=512):
    tokenizer = DistilBertTokenizer.from_pretrained(modelname)

    tokens_field = Field(
        use_vocab=False,
        preprocessing=partial(
            bert_text_preprocess,
            tokenizer=tokenizer,
            max_len=max_len
        ),
        init_token=tokenizer.cls_token_id,
        pad_token=tokenizer.pad_token_id,
        unk_token=tokenizer.unk_token_id,
    )

    tags_field = Field(
        is_target=True,
        unk_token=None,
        init_token="<pad>",
        preprocessing=partial(bert_tag_preprocessor, max_len=max_len),
    )

    fields = [
        ("tokens", tokens_field),
        ("tags", tags_field),
    ]

    return TextPreprocessor(fields, 1)


def build_bert_model():
    bert = DistilBertModel.from_pretrained("distilbert-base-cased")
    model = TaggerNet(
        module=BERTTagger,
        module__bert=bert,
        module__tagset_size=1,
        optimizer=torch.optim.Adam,
        optimizer__lr=5e-5,  # Original learning rate
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
            DynamicTagSetter(),
            skorch.callbacks.GradientNormClipping(1.),
        ],
    )

    full = make_pipeline(
        build_bert_preprocessor(),
        model,
    )
    return full


def evaluate_model(name, model, train, test):
    with timer(f"training {name}"):
        model.fit(train)

    with timer(f"evaluation on train for {name}"):
        print("F1 train: ", model.score(train))

    print("F1 test:", model.score(train))
    return model


def main():
    nltk.download('brown')
    nltk.download('universal_tagset')

    data = to_pandas(nltk.corpus.brown.tagged_sents(tagset='universal'))
    print(data.head())

    models = [
        ("base", build_model()),
        ("embeddings", build_emb_model()),
        ("bert", build_bert_model()),
    ]

    train, test = train_test_split(data)
    models = {
        name: evaluate_model(name, model, train, test)
        for name, model in models
    }


if __name__ == '__main__':
    main()
