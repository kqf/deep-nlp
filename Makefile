.PHONY: all submit

qqp = quora-question-pairs
imdb = imdb-dataset-of-50k-movie-reviews


data:
	mkdir -p data/
	curl -k -L "https://drive.google.com/uc?export=download&id=1z7avv1JiI30V4cmHJGFIfDEs9iE4SHs5" -o data/surnames.txt
	curl -k -L "https://drive.google.com/uc?export=download&id=1ji7dhr9FojPeV51dDlKRERIqr3vdZfhu" -o data/surnames-multilang.txt
	curl -k -L "https://drive.google.com/uc?export=download&id=1Pq4aklVdj-sOnQw68e1ZZ_ImMiC8IR1V" -o data/tweets.csv.zip
	curl -k -L "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab" -o data/news.zip
	curl -k -L "https://drive.google.com/uc?export=download&id=1h8dplcVzRkbrSYaTAbXYEAjcbApMxYQL" -o data/squad.zip
	unzip data/squad.zip -d data/

	curl http://www.manythings.org/anki/rus-eng.zip -o data/rus-eng.zip
	unzip data/rus-eng.zip -d data/
	python -m spacy download en

	git clone https://github.com/MiuLab/SlotGated-SLU.git
	mv SlotGated-SLU data

kdatasets:
	-kaggle competitions download -c $(qqp) -p data/
	mkdir -p data/$(qqp)
	unzip data/$(qqp) -d data/$(qqp)

	-kaggle datasets download -d lakshmi25npathi/$(imdb) -p data/
	mkdir -p data/$(imdb)
	unzip data/$(imdb) -d data/$(imdb)/

.PHONY: data
