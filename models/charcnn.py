import torch


class ConvClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, filters_count):
        super().__init__()

        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._conv = torch.nn.Conv2d(1, filters_count, (3, 1))
        self._dropout = torch.nn.Dropout(0.2)
        self._relu = torch.nn.ReLU()
        self._max_pooling = torch.nn.MaxPool2d(kernel_size=(1, 15))
        self._out_layer = torch.nn.Linear(filters_count * 15, 1, bias=False)
        model = torch.nn.Sequential(
            self.embedding,
            self._conv,
            self._dropout,
            self._relu,
            self._max_pooling,
            self._out_layer,
        )
        self.total = model

    def forward(self, inputs):
        '''
        inputs - LongTensor with shape (batch_size, max_word_len)
        outputs - FloatTensor with shape (batch_size,)
        '''
        self.batch_size = inputs[0]
        return self.total(inputs)

    def embedding(self, inputs):
        batch_size, max_len = inputs.shape[0]
        embed = self._embedding(inputs).view(batch_size, 1, max_len, -1)
        return embed

    def max_pooling(self, inputs):
        # Original inputs shape
        return inputs.reshape(self.batch_size, -1)

    def get_filters(self):
        return self._conv.weight.data.cpu().detach().numpy()


def main():
    model = ConvClassifier()
    print(model)


if __name__ == '__main__':
    main()
