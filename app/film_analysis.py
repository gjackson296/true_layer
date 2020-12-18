import logging
import pathlib

import pandas as pd

from app.constants import *

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class FilmAnalysis:
    IMDB_DATA = pathlib.Path('../data/imdb')
    WIKI_DATA = pathlib.Path('../data/wiki')

    def __init__(self):

        self.imdb = self._etl_imdb()

    def _etl_imdb(self, filename='movies_metadata.csv'):

        want_cols = {
            ID: str,
            IMDB_ID: str,
            ORIGINAL_TITLE: str,
            BUDGET: float,
            REVENUE: float
        }

        # Read raw data
        data = pd.read_csv(
            self.IMDB_DATA / filename,
            usecols=want_cols.keys()
        )[want_cols.keys()]

        logging.info(f"Read raw imdb data. Shape: {data.shape}")

        # Types - try to coerce where needed
        numeric_dtypes = [float]
        for col, want_type in want_cols.items():
            if data[col].dtype != want_type:
                try:
                    if want_type in numeric_dtypes:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        logging.info(f"Coercing col '{col}' to '{want_type}'. On error replace with NaN.")
                    else:
                        data[col] = data[col].astype(want_type)
                        logging.info(f"Coercing col '{col}' to '{want_type}'.")
                except:
                    raise ValueError(f"Unable to convert column '{col}' to data type '{want_type}'")

        # Drop duplicates
        before = len(data)
        data = data.drop_duplicates()
        after = len(data)

        logging.info(f"Removed {before-after:,} rows (from {before} to {after}) by de-duplicating entire rows")

        # Check primary key
        pkey = ID
        data.set_index(pkey, inplace=True)
        filter_dup_pkeys = data.index.duplicated()
        dup_pkeys = sorted(data.loc[filter_dup_pkeys].index.unique())
        if dup_pkeys:
            raise ValueError(f'Bad primary key. Found {len(dup_pkeys)} duplicate primary keys. '
                             f'First 5 in order are: {dup_pkeys[0:5]}')
        logging.info(f"Primary key is {pkey}")

        # Order by ID
        data = data.sort_index()
        logging.info(f"Ordering by {pkey}")

        return data

if __name__ == '__main__':
    fa = FilmAnalysis()
    foo = 1