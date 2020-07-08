import time
import torch
import skorch
import random
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve

from torchtext.data import Dataset, Example, Field, LabelField
from torchtext.data import BucketIterator


"""
!mkdir -p data
!pip3 -qq install torch
!pip install -qq bokeh
!pip install -qq eli5
!pip install scikit-learn
!curl -k -L "https://drive.google.com/uc?export=download&id=1z7avv1JiI30V4cmHJGFIfDEs9iE4SHs5" -o data/surnames.txt
"""  # noqa

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def data(filename="data/surnames.txt"):
    return pd.read_csv(filename, sep="\t", names=["surname", "label"])


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


class ConvClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, word_size=22, filters_count=3):
        super().__init__()

        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._conv = torch.nn.Conv2d(1, 1, (filters_count, 1))
        self._relu = torch.nn.ReLU()
        self._max_pooling = torch.nn.MaxPool2d(
            kernel_size=(word_size - filters_count + 1, 1))
        self._out_layer = torch.nn.Linear(emb_dim, 2, bias=False)

    def forward(self, inputs):
        '''
        inputs - LongTensor with shape (batch_size, max_word_len)
        outputs - FloatTensor with shape (batch_size,)
        '''
        outputs = self.embed(inputs)
        return self._out_layer(outputs).squeeze(1).squeeze(1)

    def embed(self, inputs):

        embs = self._embedding(inputs)
        model = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            self._conv,
            # torch.nn.Dropout(0.2),
            self._relu,
            self._max_pooling,
        )
        return model(embs.unsqueeze(dim=1))


class Tokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        chars = set("".join(X))
        self.c2i = {c: i + 1 for i, c in enumerate(chars)}
        self.c2i['<pad>'] = 0

        word_len_counter = Counter(list(map(len, X)))

        threshold = 0.99
        self.max_len = self._find_max_len(word_len_counter, threshold)
        return self

    @staticmethod
    def _find_max_len(counter, threshold):
        sum_count = sum(counter.values())
        cum_count = 0
        for i in range(max(counter)):
            cum_count += counter[i]
            if cum_count > sum_count * threshold:
                return i
        return max(counter)

    def transform(self, X):
        shorted_data = []
        for word in X:
            cc = np.array([self.c2i.get(s, 0) for s in word[:self.max_len]])
            padded = np.pad(cc, (0, self.max_len - len(cc)), mode="constant")
            shorted_data.append(padded)
        return np.array(shorted_data)


def custom_f1(y_pred, y):
    positives = y_pred.astype(bool)
    tp = np.sum(y_pred[positives] & y[positives])
    fp = np.sum(y_pred[positives] & 1 - y[positives])
    fn = np.sum(y_pred[~positives] | y[~positives])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * precision * recall / (precision + recall)
    return f1 if np.isfinite(f1) else 0


class CharClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, batch_size=32, epochs_count=1, verbose=True):
        self.batch_size = batch_size
        self.epochs_count = epochs_count
        self.verbose = verbose
        self.model = None
        self.lr = lr
        self.clsses_ = [0, 1]

    @staticmethod
    def batches(X, y, batch_size):
        num_samples = X.shape[0]

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_idx = indices[start: end]
            yield X[batch_idx], y[batch_idx]

    def fit(self, X, y):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = ConvClassifier(
            vocab_size=(np.max(X) + 1),
            emb_dim=24,
            word_size=X.shape[1],
            filters_count=3).to(device)

        optimizer = torch.optim.Adam(
            [param for param in self.model.parameters()
             if param.requires_grad],
            lr=self.lr)

        loss_function = torch.nn.CrossEntropyLoss()
        for epoch in range(self.epochs_count):
            batches = self.batches(X, y, self.batch_size)
            epoch_loss = 0
            time_epoch = time.time()
            for i, (X_batch, y_batch) in enumerate(batches):
                Xt = torch.LongTensor(X_batch).to(device)
                yt = torch.LongTensor(y_batch).to(device)

                logits = self.model.forward(Xt)
                loss = loss_function(logits, yt)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                te = time.time() - time_epoch
                print(f"Epoch {epoch}, loss {epoch_loss}, time {te:.2f}")
        return self

    def predict_proba(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.LongTensor(X).to(device)
        logits = self.model(X)
        return torch.sigmoid(logits).cpu().data.numpy()

    def embeddings(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.LongTensor(X).to(device)
        return self.model.embed(X).cpu().data.numpy()

    def predict(self, X):
        self.model.eval()
        return self.predict_proba(X).argmax(axis=1)


def build_preprocessor():
    text_field = Field(tokenize=lambda x: x)

    fields = [
        ("surname", text_field),
        ("label", LabelField(is_target=True)),
    ]
    return TextPreprocessor(fields)


def build_model(module=ConvClassifier):
    model = skorch.NeuralNet(
        module=module,
        module__vocab_size=10,  # Dummy dimension
        module__n_sentiments=2,
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
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print(f'Alloc.:{round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB')
        print(f'Cached:{round(torch.cuda.memory_cached(0) / 1024**3, 1)} GB')

    df = data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["surname"], df["label"].values, test_size=0.33, random_state=42)

    model = build_model(epochs_count=10, batch_size=2048).fit(X_tr, y_tr)

    print("F1 test", f1_score(model.predict(X_te), y_te))
    print("F1 train", f1_score(model.predict(X_tr), y_tr))

    print("Precision", precision_score(model.predict(X_te), y_te))
    print("Recall", recall_score(model.predict(X_te), y_te))
    plot_precision_recall_curve(model, X_te, y_te)

    # from models.visualize import visualize_embeddings
    word_indices = np.random.choice(
        np.arange(len(X_te)), 1000, replace=False)
    words = [X_te[ind] for ind in word_indices]
    labels = y_te[word_indices]

    words = model.steps[0][-1].transform(words)
    embeddings = model.embeddings(words)

    colors = ['red' if label else 'blue' for label in labels]
    visualize_embeddings(embeddings, words, colors)


if __name__ == '__main__':
    main()
