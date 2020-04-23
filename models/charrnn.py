import math
import torch
import numpy as np


"""
!mkdir -p data
!pip -qq install torch
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
"""  # noqa


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


class BasicRNNClassifier():
    def __init__(self,
                 hidden_size=100,
                 batch_size=100,
                 epochs_count=50,
                 print_frequency=10):

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs_count = epochs_count
        self.print_frequency = print_frequency

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.rnn = MemorizerModel(10, hidden_size=self.hidden_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.rnn.parameters())

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        batchs_count = int(math.ceil(len(X) / self.batch_size))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        total_loss = 0
        for epoch in range(self.epochs_count):
            for batch_indices in np.array_split(indices, batchs_count):
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                # Convention all RNNs: [sequence, batch, input_size]
                x_rnn = X_batch.T[:, :, np.newaxis]

                batch = torch.LongTensor(x_rnn).to(device)
                labels = torch.LongTensor(y_batch).to(device)

                optimizer.zero_grad()

                self.rnn.eval()
                logits = self.rnn(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            self._status(loss, epoch)

        return self

    def _status(self, loss, epoch=-1):
        if (epoch + 1) % self.print_frequency != 0:
            return
        self.rnn.eval()

        with torch.no_grad():
            msg = '[{}/{}] Train: {:.3f}'
            print(msg.format(
                epoch + 1,
                self.epochs_count,
                loss / self.epochs_count)
            )

    def predict_proba(self, X):
        X = np.array(X)
        self.rnn.eval()

        # Convention all RNNs: [sequence, batch, input_size]
        x_rnn = X.T[:, :, np.newaxis]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch = torch.LongTensor(x_rnn).to(device)
        with torch.no_grad():
            preds = torch.nn.functional.softmax(self.rnn(batch), dim=-1)
        return preds.detach().cpu().data.numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1)


def main():
    pass


if __name__ == '__main__':
    main()
