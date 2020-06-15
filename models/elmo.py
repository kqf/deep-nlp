
def read_dataset(path):
    data = []
    with open(path) as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line and words:
                data.append((words, tags))
                words, tags = [], []
                continue
            word, pos_tag, synt_tag, ner_tag = line.split()
            words.append(word)
            tags.append(ner_tag)
        if words:
            data.append((words, tags))
    return data[1:]


def data(dataset="train"):
    return read_dataset(f"data/conll_2003/{dataset}.txt")


def main():
    df = data()
    print(df[:3])


if __name__ == '__main__':
    main()
