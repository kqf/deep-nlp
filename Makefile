.PHONY: all submit

competition = quora-question-pairs


data:
	-kaggle competitions download -c $(competition) -p data/
	unzip data/$(competition) -d data/

	curl -k -L "https://drive.google.com/uc?export=download&id=1z7avv1JiI30V4cmHJGFIfDEs9iE4SHs5" -o data/surnames.txt
	curl -k -L "https://drive.google.com/uc?export=download&id=1ji7dhr9FojPeV51dDlKRERIqr3vdZfhu" -o data/surnames-multilang.txt
	curl -k -L "https://drive.google.com/uc?export=download&id=1Pq4aklVdj-sOnQw68e1ZZ_ImMiC8IR1V" -o data/tweets.csv.zip

.PHONY: data
