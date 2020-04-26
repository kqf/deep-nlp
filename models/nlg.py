import torch


class ConvLM(torch.nn.Module):
    def __init__(self, vocab_size, seq_length=128, emb_dim=16, window_size=5):
        super().__init__()

        padding = window_size - 1
        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._conv = torch.nn.Conv2d(1, 1, (window_size, 1),
                                     padding=(padding, 0))

        self._relu = torch.nn.ReLU()
        self._max_pooling = torch.nn.MaxPool2d(
            kernel_size=(seq_length + padding - window_size + 1, 1))
        self._out_layer = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, inputs):
        '''
        inputs - LongTensor with shape (batch_size, max_word_len)
        outputs - FloatTensor with shape (batch_size,)
        '''
        outputs = self.embed(inputs.T)
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
