# Data Engineer - Challenge v1.02

### About

App to analyse imdb movies with the highest revenue-to-budget ratio.

### How to run

1) Get data

Get and unpack wiki and imdb data from these locations: 

https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz (722Mb)
https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv (228Mb)

And place these two files in the data folder in this repo, so it should look like:

```bash
data/enwiki-latest-abstract.xml
data/movies_metadata.csv
```

2) Setup environment

To setup the environment, run:

```bash
docker-compose up
```

This will set up a postgres instance and Python3 app to allow further analysis.

Follow [these](https://docs.docker.com/compose/install/) instructions to setup `docker-compose` if you don't have it.

3) Open jupyter

If step two runs correctly, you should see a link a juptyer notebook link in the terminal, e.g.:

```bash

```

Copy-paste the link into a browser to open jupyter.

4) Analyse data

Once the jupyter notebook server is running, open and run the `analysis.ipynb`.

It takes about 15 minutes to run on a decent laptop.

This will analyse the data and store the results in postgres.

You can then use Python or SQL for further analysis, e.g.:

```python

fa.run_sql("select * from matches")

```

### Local development

To develop locally:

1) Start postgres

```bash
docker-compose up -d pg
```

2) Start jupyter

```bash
PG_HOST=localhost pipenv run jupyter notebook
```

### Decisions

- `lxml` is faster and more efficient when parsing xml than `elementtree`

- `TfidfVectorizer` is [faster](https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536) than standard 
fuzzy matching techniques.

- `docker-compose` makes results easier to reproduce on other machines.

### TODO

- Fix fuzzy matching, cf:
```bash
https://en.wikipedia.org/wiki/Apollo_13
vs
https://en.wikipedia.org/wiki/Apollo_13_(film)
```

- Persist db results to volume

- Add pytests

- Add threshold for revenue to budget ratios; e.g. minimum budge of 10,000 dollars.
