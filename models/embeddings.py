from torchtext.data import Dataset, Example
from torchtext.data import Field


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


def build_preprocessor():
    pass


def main():
    word = Field(tokenize=lambda x: [x], batch_first=True)
    fields = [
        ('context', word),
        ('target', word)
    ]
    raw = [
        "first sentence",
        "second sentence",
    ]

    data = SkipGramDataset(raw, fields)
    word.build_vocab(data)


if __name__ == '__main__':
    main()
