.PHONY: all submit

competition = quora-question-pairs


data:
	-kaggle competitions download -c $(competition) -p data/
	unzip data/$(competition) -d data/

.PHONY: data
