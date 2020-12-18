import logging
import pathlib
from lxml import etree
import gc
import numpy as np
import pandas as pd
from nltk import bigrams
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def run():


    # Question 2a
    meta_data['ratio'] = meta_data[REVENUE] / meta_data[BUDGET]
    meta_data['ratio'] = meta_data['ratio'].replace([np.inf, -np.inf], np.nan)

    # Question 2b
    #import xml.etree.ElementTree as ET
    #tree = ET.parse(WIKI_DATA / 'enwiki-latest-abstract.xml')
    # TODO pre-cache results if it's slow?

    # ITERPARSE METHOD
    # TODO investigate Parser Target alternative
    # get an iterable
    context = etree.iterparse(
        source=str(WIKI_DATA / 'enwiki-latest-abstract.xml'),
        # If you let all tags through then:
        #   PRO: you can clear the element to save memory
        #   CON: clearing elements means the process is about twice as long
        #tag=['title', 'url', 'abstract']
    )

    # TODO delete?
    # # turn it into an iterator
    # context = iter(context)
    #
    # # get the root element
    # event, root = context.__next__()

    titles, urls, abstracts = [], [], []
    for event, elem in context:
        if elem.tag == 'title':
            titles.append(elem.text)
        elif elem.tag == 'url':
            urls.append(elem.text)
        elif elem.tag == 'abstract':
            abstracts.append(elem.text)

            if len(urls) > 0 and len(urls) % 10_000 == 0:
                logging.info(f'len {len(urls)} cf {len(urls) % 100_000}')
                # TODO remove tmp limit
                break

        else:

            elem.clear(keep_tail=True)

    results = pd.DataFrame.from_dict({
        'title': titles,
        'url': urls,
        'abstracts': abstracts
    })

    # Tidy
    results['title'] = results['title'].str.strip('Wikipedia: ')

    # Cache
    results.to_parquet('/tmp/imdb_data.parquet')

    # MATCHING
    # Have: imdb and wiki

    # https://bergvca.github.io/2017/10/14/super-fast-string-matching.html
    import re

    def ngrams(string, n=5):
        string = re.sub(r'[,-./]|\sBD', r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    from sklearn.feature_extraction.text import TfidfVectorizer

    imdb_titles = meta_data.loc[:, ORIGINAL_TITLE].tolist()
    wiki_titles = results.loc[:, 'title'].tolist()

    #imdb_titles = ['some test value 1', 'another one']
    #wiki_titles = ['completely another one']

    cvect = CountVectorizer(analyzer='word')
    vocabulary = cvect.fit(imdb_titles + wiki_titles).vocabulary_

    vectorizer = TfidfVectorizer(min_df=1, vocabulary=vocabulary) #, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(imdb_titles)
    tf_idf_matrix2 = vectorizer.fit_transform(wiki_titles)

    from scipy.sparse import csr_matrix
    import sparse_dot_topn.sparse_dot_topn as ct

    def awesome_cossim_top(A, B, ntop, lower_bound=0):
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M * ntop

        indptr = np.zeros(M + 1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

        return csr_matrix((data, indices, indptr), shape=(M, N))

    matches = awesome_cossim_top(tf_idf_matrix2, tf_idf_matrix.transpose(), 1, 0.95)

    # To join together
    wiki_idx, imdb_idx = matches.nonzero()

    meta_data['prob_wiki_idx'] = np.nan
    meta_data.iloc[imdb_idx, 4] = wiki_idx

    output = meta_data.set_index('prob_wiki_idx').join(results, how='left')

    # # TODO don't need?
    # def get_matches_df(sparse_matrix, name_vector1, name_vector2, top=100):
    #     non_zeros = sparse_matrix.nonzero()
    #
    #     sparserows = non_zeros[0]
    #     sparsecols = non_zeros[1]
    #
    #     if top:
    #         nr_matches = top
    #     else:
    #         nr_matches = sparsecols.size
    #
    #     left_side = np.empty([nr_matches], dtype=object)
    #     right_side = np.empty([nr_matches], dtype=object)
    #     similairity = np.zeros(nr_matches)
    #
    #     for index in range(0, nr_matches):
    #         left_side[index] = name_vector1[sparserows[index]]
    #         right_side[index] = name_vector2[sparsecols[index]]
    #         similairity[index] = sparse_matrix.data[index]
    #
    #     return pd.DataFrame({'left_side': left_side,
    #                          'right_side': right_side,
    #                          'similairity': similairity})
    #
    # matches_df = get_matches_df(matches, wiki_titles, imdb_titles, top=100)
    # #matches_df = get_matches_df(matches, list(vocabulary.keys()), top=100)
    #
    # # Don't need this since we have two different lists
    # #matches_df = matches_df[matches_df['similairity'] < 0.99999]  # Remove all exact matches
    # matches_df.sample(20)

    # TODO apollo 13 vs apollo 13 film?
    foo = 1

    # Get top movies
    # TODO I assumed best investments; the question wording confused me
    #  There are obvious dq issues here! Maybe put a lower limit of 1k?
    top_n = output.sort_values(by='ratio', ascending=False).head(1_000)


    # TODO can we make the above faster?
    #  I think we should cache the url list.
    #  How can we match names and urls?
    #   ...efficiently!
    #  ARE YOU FINDING TOO MANY URLS!? CAN YOU ONLY LOOK FOR ONES AFTER TITLE?
    #  ***Memory hitting 20Gb

    # You generate a list of URLS in 10 minutes
    # BUT - are the URLs unique? Can you look up based on URL?
    #   ...Or should you be caching BOTH URL and abstract somewhere?

    # Cache abstracts at the same time.
    # Store as HDFS and have the ability to update cache

    # Then you want a db setup; just use the sqlite

    # Add argparse for each bit



    # lxml seems to be faster than element tree; and use less memory !?

if __name__ == '__main__':
    run()
