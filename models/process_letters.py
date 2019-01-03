import cfg
import pandas as pd
from nltk.corpus import stopwords


def clean_docs(letters_df):
    letters_df['clean_doc'] = letters_df['letter_text'].str.replace("[^a-zA-Z#]", " ")
    letters_df['clean_doc'] = letters_df['clean_doc'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    letters_df['clean_doc'] = letters_df['clean_doc'].apply(lambda x: x.lower())
    stop_words = stopwords.words('english')

    # tokenization
    tokenized_doc = letters_df['clean_doc'].apply(lambda x: x.split())

    # remove stop words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

    # de-tokenization
    detokenized_doc = []
    for i in range(len(letters_df)):
        t = ' '.join(tokenized_doc.iloc[i])
        detokenized_doc.append(t)

    letters_df['clean_doc'] = detokenized_doc

    # save processed df
    letters_df.to_parquet(cfg.proc_letters_df_parq_path,compression='gzip')

    return letters_df