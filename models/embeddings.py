import torch
import skorch
import numpy as np

from torchtext.data import Dataset, Example
from torchtext.data import Field, BucketIterator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


def split(sentence):
    return sentence.split()


class SkipGramDataset(Dataset):

    def __init__(self, lines, fields, tokenize=split, window_size=3, **kwargs):
        examples = []
        ws = window_size
        for line in lines:
            words = tokenize(line.strip())
            if len(words) < window_size + 1:
                continue

            for i in range(len(words)):
                contexts = words[max(0, i - ws):i]
                contexts += words[
                    min(i + 1, len(words)):
                    min(len(words), i + ws) + 1
                ]

                for context in contexts:
                    examples.append(Example.fromlist(
                        (context, words[i]), fields))
        super(SkipGramDataset, self).__init__(examples, fields, **kwargs)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, dtype=SkipGramDataset):
        self.fields = fields
        self.dtype = dtype

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            field.build_vocab(dataset)
        return self

    def transform(self, X, y=None):
        return self.dtype(X, self.fields)


def build_preprocessor():
    word = Field(tokenize=lambda x: [x], batch_first=True)
    fields = [
        ('context', word),
        ('target', word)
    ]
    return TextPreprocessor(fields, dtype=SkipGramDataset)


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.context.view(-1), batch.target.view(-1)


class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size=1, embedding_dim=100):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        latent = self.embeddings(context)
        return self.out_layer(latent)


class NegativeSamplingIterator(BucketIterator):
    def __init__(self, dataset, batch_size,
                 neg_samples, ns_exponent, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.ns_exponent = ns_exponent
        self.neg_samples = neg_samples

        vocab = dataset.fields["context"].vocab
        freq = [vocab.freqs[s]**self.ns_exponent for s in vocab.itos]

        # Normalize
        self.freq = np.array(freq) / np.sum(freq)

    def __iter__(self):
        for batch in super().__iter__():
            inputs = {
                "context": batch.context.view(-1),
                "target": batch.target.view(-1),
                "negatives": self.sample(batch.context),
            }
            yield inputs, torch.empty(0)

    def sample(self, context):
        negatives = np.random.choice(
            np.arange(len(self.freq)),
            p=self.freq,
            size=(context.shape[0], self.neg_samples),
        )
        return torch.tensor(negatives, dtype=context.dtype).to(context.device)


class SGNSModel(torch.nn.Module):
    def __init__(self, vocab_size=1, embedding_dim=100):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_v = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context, target, negatives):
        # u[batch_size, embedding_dim]
        u = self.embeddings(context)
        # v[batch_size, embedding_dim]
        v = self.embeddings_v(target)
        # vp[batch_size, neg_samples, embedding_dim]
        vp = self.embeddings_v(negatives)

        # log(sigma(sum_{i = emb_dim} u_i * v_i)) -> [batch_size]
        pos = torch.nn.functional.logsigmoid((v * u).sum(-1))

        # sum_{i = emb_dim} -vp_i * u_i) -> [batch_size, neg_samples]
        neg_sim = (-vp * u.unsqueeze(dim=1)).sum(-1)

        # sum_{neg_samples} log(sigma(neg_sim_i)) -> [batch_size]
        neg = torch.nn.functional.logsigmoid(neg_sim).sum(-1)

        # Calculate loss
        # logsigmoid(v * u) + sum_{neg} logsigmoid(vp * u) -> [batch_size]
        return -(pos + neg).mean()


class DynamicParameterSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["context"].vocab
        net.set_params(module__vocab_size=len(vocab))

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model():
    model = skorch.NeuralNet(
        module=SkipGramModel,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=100,
        iterator_train=SkorchBucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SkorchBucketIterator,
        iterator_valid__shuffle=True,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            DynamicParameterSetter(),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


class SGNSLanguageModel(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        return y_pred


def build_sgns_model():
    model = SGNSLanguageModel(
        module=SGNSModel,
        optimizer=torch.optim.Adam,
        criterion=lambda: None,  # Does nothing
        max_epochs=2,
        batch_size=128,
        iterator_train=NegativeSamplingIterator,
        iterator_train__neg_samples=5,
        iterator_train__ns_exponent=3. / 4.,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=NegativeSamplingIterator,
        iterator_valid__neg_samples=5,
        iterator_valid__ns_exponent=3. / 4.,
        iterator_valid__shuffle=True,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            DynamicParameterSetter(),
            skorch.callbacks.Initializer('*',
                fn=torch.nn.init.xavier_normal_),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    raw = [
        "first sentence",
        "second sentence",
    ]
    data = build_preprocessor().fit_transform(raw)
    print(data)


if __name__ == '__main__':
    main()
