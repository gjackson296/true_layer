import logging
import os
import pathlib

import numpy as np
import pandas as pd
from lxml import etree
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.constants import *
from app.utils import awesome_cossim_top

logging.basicConfig(format='%(asctime)s %(name)s %(message)s', level=logging.INFO)


class FilmAnalysis:
    DEFAULT_DATA_FOLDER = pathlib.Path('data')

    def __init__(self, db_user, db_pass, db_host, db_name, db_port, data_folder = None):

        self.engine = create_engine(f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}')
        self.data_folder = data_folder or self.DEFAULT_DATA_FOLDER
        self.imdb = self._etl_imdb()
        #self.wiki = self._etl_wiki()

    def _etl_imdb(self, filepath='movies_metadata.csv', top_n=1_000):
        """
        Load and clean imdb data
        """

        logging.info("LOAD IMDB")

        want_cols = {
            IMDB_ID: int,
            IMDB_ID2: str,
            IMDB_ORIGINAL_TITLE: str,
            IMDB_BUDGET: float,
            IMDB_REVENUE: float,
            IMDB_PROD_COMP: str
        }

        # Read raw data
        data = pd.read_csv(self.data_folder / filepath, usecols=want_cols.keys())[want_cols.keys()]

        logging.info(f"Read raw imdb data. Shape: {data.shape}")

        # Types - try to coerce where needed
        numeric_dtypes = [int, float]
        for col, want_type in want_cols.items():
            if data[col].dtype != want_type:
                try:
                    if want_type in numeric_dtypes:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        logging.info(f"Coercing col '{col}' to '{want_type}'. On error replace with NaN.")
                    else:
                        data[col] = data[col].astype(want_type)
                        logging.info(f"Coercing col '{col}' to '{want_type}'.")
                except Exception as e:
                    raise ValueError(f"Unable to convert column '{col}' to data type '{want_type}'")\
                        .with_traceback(e.__traceback__)

        # Drop duplicates
        before = len(data)
        data = data.drop_duplicates()
        after = len(data)
        logging.info(f"Removed {before-after:,} rows (from {before:,} to {after:,}) by de-duplicating entire rows.")

        # Drop nan ids
        before = len(data)
        filter_id_is_nan = data[IMDB_ID].isna()
        data = data.loc[~filter_id_is_nan]
        after = len(data)
        logging.info(f"Removed {before-after:,} rows (from {before:,} to {after:,}) by removing NaN '{IMDB_ID}'s.")

        # Check primary key
        pkey = IMDB_ID
        data.set_index(pkey, inplace=True)
        filter_dup_pkeys = data.index.duplicated()
        dup_pkeys = sorted(data.loc[filter_dup_pkeys].index.unique())
        if dup_pkeys:
            raise ValueError(f'Bad primary key. Found {len(dup_pkeys)} duplicate primary keys. '
                             f'First 5 in order are: {dup_pkeys[0:5]}')
        logging.info(f"Primary key is '{pkey}'.")

        # Order by IMDB_ID
        data = data.sort_index()
        logging.info(f"Order by {pkey}.")

        # Add ratio
        col = f'ratio_{IMDB_REVENUE}_to_{IMDB_BUDGET}'
        data[col] = data[IMDB_REVENUE] / data[IMDB_BUDGET]
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        logging.info(f"Add '{col}' (higher means the film performed better).")

        # Keep top n
        data = data.sort_values(by=col, ascending=False).head(top_n)
        logging.info(f"Keep top {top_n:,} by '{col}'")

        return data


    def _etl_wiki(self, filename='enwiki-latest-abstract.xml'):
        """
        Load wiki data

        Note on reducing memory footprint:
            We don't pass a 'tag' arg to iterparse(). If we did, the 'elem.clear(keep_tail=True)'
            path below would never be reached, and so lots of tree elements we don't need would be parsed.
            TODO investigate Parser Target alternative approach which use less CPU/RAM.
        """

        logging.info("LOAD WIKI")

        context = etree.iterparse(source=str(self.data_folder / filename))

        # TODO count URL so you can show progress? Cache results?

        # Cache the data we need
        titles, urls, abstracts = [], [], []
        counter = 0
        for event, elem in context:

            if elem.tag == 'title':
                titles.append(elem.text)

                # Give an idea of progress
                if len(titles) % 500_000 == 0:
                    logging.info(f"Processed {len(titles):,} entries.")

            elif elem.tag == 'url':
                urls.append(elem.text)
            elif elem.tag == 'abstract':
                abstracts.append(elem.text)

            else:

                elem.clear(keep_tail=True)

        # Prepare output
        data = pd.DataFrame.from_dict({
            WIKI_TITLE: titles,
            WIKI_URL: urls,
            WIKI_ABSTRACT: abstracts
        })

        # Tidy
        prefix = 'Wikipedia: '
        data[WIKI_TITLE] = data[WIKI_TITLE].str.strip(prefix)
        logging.info(f"Remove prefix '{prefix}' from '{WIKI_TITLE}' column.")

        return data

    def match_imdb_to_wiki(self):
        """
        Match wiki data to imdb data

        Use fast term-frequency inverse-document-frequency approach as per:
            https://bergvca.github.io/2017/10/14/super-fast-string-matching.html
        """

        logging.info("MATCH WIKI TO IMDB")

        imdb_titles = self.imdb.loc[:, IMDB_ORIGINAL_TITLE].tolist()
        wiki_titles = self.wiki.loc[:, WIKI_TITLE].tolist()

        cvect = CountVectorizer(analyzer='word')
        vocabulary = cvect.fit(imdb_titles + wiki_titles).vocabulary_

        vectorizer = TfidfVectorizer(min_df=1, vocabulary=vocabulary)
        imdb_tf_idf_matrix = vectorizer.fit_transform(imdb_titles)
        wiki_tf_idf_matrix = vectorizer.fit_transform(wiki_titles)

        matches = awesome_cossim_top(wiki_tf_idf_matrix, imdb_tf_idf_matrix.transpose(), 1, 0.95)

        # To join together
        wiki_idx, imdb_idx = matches.nonzero()

        # Prepare output
        # Start with imdb data
        matched = self.imdb.copy()
        # Add the index of the nominal wiki match
        col = 'nominal_wiki_idx'
        matched[col] = np.nan
        col_idx = matched.columns.get_loc(col)
        matched.iloc[imdb_idx, col_idx] = wiki_idx
        # Add wiki information
        matched = matched.set_index(col).join(self.wiki.drop(columns=WIKI_TITLE), how='left')

        return matched

    def write_matches_to_pg(self, matches, if_exists='fail'):

        with self.engine.connect() as conn:
            matches.to_sql('matches', conn, if_exists=if_exists, index_label=IMDB_ID)

        logging.info(f"Wrote {len(matches):,} to postgres.")


    def run_sql(self, sql):

        with self.engine.connect() as conn:
            data = pd.read_sql(sql, conn)

        logging.info(f"Read {len(data):,} from postgres.")

        return data
