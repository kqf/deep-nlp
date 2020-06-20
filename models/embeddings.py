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


class NegativeSamplingIterator(BucketIterator):
    def __init__(self, neg_samples, ns_exponent, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
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


class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size=1, embedding_dim=100):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        latent = self.embeddings(context)
        return self.out_layer(latent)


class SGNSModel(torch.nn.Module):
    def __init__(self, vocab_size=1, embedding_dim=100):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_v = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs, targets, negatives):
        u = self.embeddings(inputs)
        v = self.embeddings_v(targets)
        vp = self.embeddings_v(negatives)

        pos = torch.nn.functional.logsigmoid((v * u).sum(1))

        # vp[batch, neg, v] * u[batch, v] -> [batch, neg, 1] -> [barch, neg]
        neg_prod = torch.bmm(vp, u.unsqueeze(dim=2)).squeeze()

        neg = torch.nn.functional.logsigmoid(torch.sum(neg_prod, dim=1))
        return -(pos - neg).mean()


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


def main():
    raw = [
        "first sentence",
        "second sentence",
    ]
    data = build_preprocessor().fit_transform(raw)
    print(data)


if __name__ == '__main__':
    main()
