import logging
import pathlib

import numpy as np
import pandas as pd
from lxml import etree

from app.constants import *

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class FilmAnalysis:
    IMDB_DATA = pathlib.Path('../data/imdb')
    WIKI_DATA = pathlib.Path('../data/wiki')

    def __init__(self):

        self.imdb = self._etl_imdb()
        self.wiki = self._etl_wiki()

    def _etl_imdb(self, filename='movies_metadata.csv'):
        """
        Load and clean imdb data
        """

        logging.info("LOAD IMDB")

        want_cols = {
            IMDB_ID: int,
            IMDB_ID2: str,
            IMDB_ORIGINAL_TITLE: str,
            IMDB_BUDGET: float,
            IMDB_REVENUE: float
        }

        # Read raw data
        data = pd.read_csv(
            self.IMDB_DATA / filename,
            usecols=want_cols.keys()
        )[want_cols.keys()]

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
        logging.info(f"Removed {before-after:,} rows (from {before} to {after}) by de-duplicating entire rows.")

        # Drop nan ids
        before = len(data)
        filter_id_is_nan = data[IMDB_ID].isna()
        data = data.loc[~filter_id_is_nan]
        after = len(data)
        logging.info(f"Removed {before-after:,} rows (from {before} to {after}) by removing NaN '{IMDB_ID}'s.")

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

        return data


    def _etl_wiki(self):
        """
        Load wiki data

        Note on reducing memory footprint:
            We don't pass a 'tag' arg to iterparse(). If we did, the 'elem.clear(keep_tail=True)'
            path below would never be reached, and so lots of tree elements we don't need would be parsed.
            TODO investigate Parser Target alternative approach which use less CPU/RAM.
        """

        logging.info("LOAD WIKI")

        context = etree.iterparse(source=str(self.WIKI_DATA / 'enwiki-latest-abstract.xml'))

        # TODO count URL so you can show progress? Cache results?

        # Cache the data we need
        titles, urls, abstracts = [], [], []
        counter = 0
        for event, elem in context:

            # Give an idea of progress
            if len(titles) % 100_000 == 0:
                logging.info(f"Processed {len(titles)} entries.")

            if elem.tag == 'title':
                titles.append(elem.text)
            elif elem.tag == 'url':
                urls.append(elem.text)
            elif elem.tag == 'abstract':
                abstracts.append(elem.text)

                # TODO remove tmp limit
                if len(urls) > 0 and len(urls) % 10_000 == 0:
                    logging.info(f'len {len(urls)} cf {len(urls) % 100_000}')
                    break

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

if __name__ == '__main__':
    fa = FilmAnalysis()
    foo = 1