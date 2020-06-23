import torch
import skorch
import random
import numpy as np
import pandas as pd

from torchtext.data import Field, LabelField, Dataset, Example
from torchtext.data import BucketIterator

from transformers import DistilBertTokenizer, DistilBertModel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

"""
!pip install -qq torch torchtext skorch numpy pandas transformers
!pip install -qq sklearn kaggle
!kaggle competitions download -c quora-question-pairs -p data/
!unzip data/quora-question-pairs -d data/
"""
SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def data(dataset="train"):
    df = pd.read_csv(f"data/{dataset}.csv.zip")
    df.replace(np.nan, '', regex=True, inplace=True)
    return df


class MergeColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col1, col2, out_col, sep_token):
        self.col1 = col1
        self.col2 = col2
        self.out_col = out_col
        self.sep_token = f" {sep_token} "

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.out_col] = X[self.col1] + self.sep_token + X[self.col2]
        return X


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, pair_field=None, min_freq=0):
        self.fields = fields
        self.pair_field = pair_field
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


class BertDoubleField(Field):
    def __init__(self, btokenizer, max_len, *args, **kwargs):
        super().__init__(tokenize=self._trim_tokenize, *args, **kwargs)
        self.btokenizer = btokenizer
        self.max_len = max_len

    def _trim_tokenize(self, sentence):
        tokens = self.btokenizer.tokenize(sentence)
        # init_token + eos_token + eos_token = 3
        tokens = tokens[:self.max_len - 3]
        return self.btokenizer.convert_tokens_to_ids(tokens)


def build_preprocessor(modelname='distilbert-base-cased', max_len=512):
    tokenizer = DistilBertTokenizer.from_pretrained(modelname)

    text_field = Field(
        batch_first=True
    )

    text_pair = BertDoubleField(
        btokenizer=tokenizer,
        max_len=max_len,
        batch_first=True,
        use_vocab=False,
        init_token=tokenizer.cls_token_id,
        eos_token=tokenizer.sep_token_id,
        pad_token=tokenizer.pad_token_id,
        unk_token=tokenizer.unk_token_id,
    )

    fields = [
        ('question1', text_field),
        ('question2', text_field),
        ('question_pair', text_pair),
        ('is_duplicate', LabelField(dtype=torch.long)),
    ]

    preprocess = make_pipeline(
        MergeColumnTransformer(
            "question1",
            "question2",
            "question_pair",
            sep_token=tokenizer.sep_token
        ),
        TextPreprocessor(fields, text_pair, min_freq=3),
    )
    return preprocess


class BaselineModel(torch.nn.Module):
    def __init__(self, vocab_size=1, n_classes=1, pad_idx=0,
                 emb_dim=100, hidden_dim=256):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim, pad_idx)
        self._rnn = torch.nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self._out = torch.nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, inputs):
        question1, question2, question_pair = inputs
        q1, _ = self._rnn(self._emb(question1))[1]
        q2, _ = self._rnn(self._emb(question2))[1]

        hidden = torch.cat([q1.squeeze(0), q2.squeeze(0)], dim=-1)
        output = self._out(hidden)

        return output


class BERTSimilarity(torch.nn.Module):
    def __init__(self, bert, vocab_size=None, n_classes=1, pad_idx=None):
        super().__init__()
        self._bert = bert
        hidden_dim = bert.config.to_dict()['dim']
        self._out = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, inputs):
        question1, question2, question_pair = inputs
        with torch.no_grad():
            hidden = self._bert(question_pair)[0].mean(dim=1)
        return self._out(hidden)


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["question1"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(module__pad_idx=vocab.stoi["<pad>"])
        net.set_params(module__n_classes=len(X.fields["is_duplicate"].vocab))


class TrainableCounter(skorch.callbacks.Callback):
    def on_train_end(self, net, X, y):
        model = net.module_
        n_pars = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'The model has {n_pars:,} trainable parameters')


class DeduplicationNet(skorch.NeuralNet):
    def predict(self, X):
        vocab = X.fields["is_duplicate"].vocab
        return np.take(vocab.itos, self.predict_proba(X).argmax(-1))


def build_model():
    bert = DistilBertModel.from_pretrained('distilbert-base-cased')
    model = DeduplicationNet(
        module=BERTSimilarity,
        module__bert=bert,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        max_epochs=2,
        batch_size=64,
        iterator_train=BucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=BucketIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            DynamicVariablesSetter(),
            skorch.callbacks.Freezer(['_bert.*']),
            TrainableCounter(),
            # skorch.callbacks.ProgressBar(),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    train = data()
    print(train.head())
    model = build_model().fit(train[:1000])

    train["pred"] = model.predict(train)
    print("Train F1:", f1_score(train["is_duplicate"], train["pred"]))

    test = data("test")
    print(test.head())
    test["pred"] = model.predict(test)
    print("Test  F1:", f1_score(test["is_duplicate"], test["pred"]))

    q8bert = torch.quantization.quantize_dynamic(
        module.module_._bert, {torch.nn.Linear}, dtype=torch.qint8
    )

    module.module_._bert = q8bert
    test["pred"] = model.predict(test)
    print("TestQ F1:", f1_score(test["is_duplicate"], test["pred"]))


if __name__ == '__main__':
    main()
