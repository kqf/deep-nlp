.PHONY: all submit

competition = quora-question-pairs


data:
	-kaggle competitions download -c $(competition) -p data/
	unzip data/$(competition) -d data/

	curl -k -L "https://drive.google.com/uc?export=download&id=1z7avv1JiI30V4cmHJGFIfDEs9iE4SHs5" -o data/surnames.txt

.PHONY: data
