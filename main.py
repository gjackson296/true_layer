import pathlib

import pandas as pd

IMDB_DATA = pathlib.Path('data/imdb')

ID = 'id'
IMDB_ID = 'imdb_id'
ORIGINAL_TITLE = 'original_title'
BUDGET = 'budget'
REVENUE = 'revenue'

def run():

    want_cols = {
        ID: str,
        IMDB_ID: str,
        ORIGINAL_TITLE: str,
        BUDGET: float,
        REVENUE: float
    }

    meta_data = pd.read_csv(
        IMDB_DATA / 'movies_metadata.csv',
        usecols=want_cols.keys()
    )[want_cols.keys()]

    # Types
    NUMERIC_DTYPES = [float]
    for col, want_type in want_cols.items():
        if meta_data[col].dtype != want_type:
            try:
                # Try to coerce
                if want_type in NUMERIC_DTYPES:
                    meta_data[col] = pd.to_numeric(meta_data['budget'], errors='coerce')

                    # TODO add logging: 'invalid parsing will be set as NaN.'
                else:
                    meta_data[col] = meta_data[col].astype(want_type)
            except:
                raise ValueError(f"Unable to convert column '{col}' to data type '{want_type}'")

    pkey = ID
    meta_data.set_index(pkey, inplace=True)

    # Drop duplicates
    meta_data = meta_data.drop_duplicates()

    # validation
    # pkey
    filter_dup_pkeys = meta_data.index.duplicated()
    dup_pkeys = sorted(meta_data.loc[filter_dup_pkeys].index.unique())
    if dup_pkeys:
        raise ValueError(f'Bad primary key. Found {len(dup_pkeys)} duplicate primary keys. '
                         f'First 5 in order are: {dup_pkeys[0:5]}')

    # Question 2a
    ratio = meta_data[BUDGET]/meta_data[REVENUE]

    # Question 2b
    import xml.etree.ElementTree as ET

    foo = 1

if __name__ == '__main__':
    run()
