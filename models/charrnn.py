import tqdm
import torch
import skorch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from torchtext.data import BucketIterator

sns.set()


"""
!mkdir -p data
!pip -qq install torch
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
"""  # noqa

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class SimpleRNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super().__init__()

        self._hidden_size = hidden_size
        # Convention: X[batch, inputs] * W[inputs, outputs]
        self._hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self._activate = activation or torch.nn.Tanh()

    def forward(self, inputs, hidden=None):
        # RNN Convention: X[sequence, batch, inputs]
        seq_len, batch_size = inputs.shape[:2]

        if hidden is None:
            hidden = inputs.new_zeros((batch_size, self._hidden_size))

        for i in range(seq_len):
            layer_input = torch.cat((hidden, inputs[i]), dim=1)
            hidden = self._activate(self._hidden(layer_input))
        return hidden


class MemorizerModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, activation=None):
        super().__init__()

        self._hidden_size = hidden_size
        self._embedding = torch.nn.Embedding.from_pretrained(
            torch.eye(embedding_size, requires_grad=True))
        self._rnn = SimpleRNNModel(embedding_size, hidden_size, activation)
        # We have as many output classes as the input ones
        output_size = embedding_size
        self._linear = torch.nn.Linear(hidden_size, output_size)

        self._model = torch.nn.Sequential(
            self._embedding,
            self._rnn,
            self._linear

        )

    def forward(self, inputs, hidden=None):
        # Convention: inputs[sequence, batch, input_size=1]
        return self._model(inputs.squeeze(dim=-1))

    def parameters(self, learnable=False):
        parameters = super().parameters()
        if learnable:
            return filter(lambda p: p.requires_grad, parameters)
        return parameters


def generate_data(num_batches=10, batch_size=100, seq_len=5):
    for _ in range(num_batches * batch_size):
        data = np.random.randint(0, 10, seq_len)
        yield data, data[0]


class SequenceGenerator(BucketIterator):
    def __init__(self, dataset, batch_size, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = kwargs.get("device", "cpu")

    def __iter__(self):
        for batch, _ in self.dataset:
            batch = torch.randint(0, 10, (batch, self.batch_size))
            batch = batch.to(self.device)
            yield batch, batch[0, :]


def build_model(module=MemorizerModel):
    model = skorch.NeuralNet(
        module=module,
        # module__vocab_size=10,  # Dummy dimension
        module__embedding_size=24,
        module__hidden_size=30,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.01,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=32,
        iterator_train=SequenceGenerator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SequenceGenerator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: (x, x),
        callbacks=[
            skorch.callbacks.GradientNormClipping(1.),
        ],
    )
    return model


def main():
    sequence_lengths = np.arange(1, 50)
    train, test = [], []
    for seq in tqdm.tqdm(sequence_lengths):
        # TODO: Add f1 score
        model = build_model.fit([seq] * 100)
        train.append(model.history[-1]["train_loss"])
        test.append(model.history[-1]["valid_loss"])

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, sharex=True)

    ax1.plot(sequence_lengths, train)
    ax2.plot(sequence_lengths, test)

    ax1.set_ylabel("f1 score")
    ax1.set_xlabel("sequence length")
    ax2.set_ylabel("f1 score")
    ax2.set_xlabel("sequence length")
    plt.show()


if __name__ == '__main__':
    main()
