import cfg
import heapq
import pandas as pd
from math import log
from operator import itemgetter
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



def build_tf_idf(proc_letters_clean_ser):
    na_list = cfg.na_list
    corpus = pd.read_csv(cfg.fin_word_corpus, keep_default_na=False, na_values=na_list)
    # Todo: Add Buffett letters to corpus to extend coverage

    # lemmatize Clean Series
    letters_clean_list_ser = proc_letters_clean_ser.apply(lambda x: x.split(' '))

    # lemmatize words
    letters_clean_list_ser = build_lemmatized_lists(letters_clean_list_ser)

    # remove words common to letters
    letters_clean_list_ser = letters_clean_list_ser.apply(lambda row: [token for token in row if token not in cfg.rem_list])

    # remove short words
    letters_clean_list_ser = letters_clean_list_ser.apply(lambda row: [token for token in row if len(token)>3])

    # remove stop words
    letters_clean_list_ser = letters_clean_list_ser.apply(lambda row: [token for token in row if token not in stop_words])

    # remove words that appear too frequently or not frequently enough
    flatten = lambda l: [item for sublist in l for item in sublist]
    letters_clean_list = list(letters_clean_list_ser)
    flat_list = flatten(letters_clean_list)
    most_freq = Counter(flat_list).most_common(int(0.05*len(flat_list)))
    least_freq = least_common_values(flat_list,int(0.05*len(flat_list)))
    letters_clean_list_ser = letters_clean_list_ser.apply(
        lambda row: [token for token in row if token not in most_freq])
    letters_clean_list_ser = letters_clean_list_ser.apply(
        lambda row: [token for token in row if token not in least_freq])

    total_word_ser = letters_clean_list_ser.apply(lambda x: len(x))

    letters_lemmatized_df = pd.concat([pd.DataFrame(letters_clean_list_ser),pd.DataFrame(total_word_ser)], axis=1)
    letters_lemmatized_df.columns = ['lemma_list','total_word_count']

    # build tf
    letters_lemmatized_df['dict_count'] = letters_lemmatized_df.apply(lambda row: dict(Counter(row.lemma_list)), axis=1)
    letters_lemmatized_df['tf'] = letters_lemmatized_df.apply(lambda row: {k: v / row.total_word_count for k, v in
                                                           row.dict_count.items()},axis=1)

    # Stem corpus and build idf dictionary
    corpus['Word'] = corpus['Word'].apply(lambda x: x.lower())
    corpus['lemma'] = corpus.apply(lambda row: get_lemma(row.Word), axis=1)
    corpus_groupby_lemma_df = pd.DataFrame(corpus.groupby('lemma')['Doc Count'].sum())
    corpus_groupby_lemma_df.columns = ['doc_count']
    corpus_groupby_lemma_df['idf_ratio'] = corpus_groupby_lemma_df['doc_count'].apply(lambda x: corpus['Doc Count'].sum()/x
                                                                                if x > 0 else 1)
    corpus_groupby_lemma_df['idf'] = corpus_groupby_lemma_df['idf_ratio'].apply(lambda x: log(x))
    idf_dict = pd.Series(corpus_groupby_lemma_df.idf.values,index=corpus_groupby_lemma_df.index).to_dict()
    tf_idf = letters_lemmatized_df.tf.apply(lambda x: {k:build_tf_idf_value(k,v, idf_dict) for k,v in x.items()})
    list_of_year_frames = [pd.DataFrame.from_dict(tf_idf.iloc[i],orient='index') for i in range(len(tf_idf))]
    for i in range(len(tf_idf)):
        list_of_year_frames[i].columns = [tf_idf.index[i]]
    tf_idf_df = pd.concat(list_of_year_frames,axis=1,sort=True).T
    tf_idf_df = tf_idf_df.fillna(0)

    # reduce dimensionality of tf_idf_df to only account for the n most frequent words across the entire corpus
    most_freq_word_keys = get_most_frequent_word_keys_across_documents(letters_lemmatized_df['lemma_list'])
    tf_idf_df_reduced = tf_idf_df[most_freq_word_keys]
    tf_idf_df_reduced.to_parquet(cfg.tf_idf_df_path,compression='gzip')
    return tf_idf_df_reduced

def get_most_frequent_word_keys_across_documents(ser_of_lists):
    big_list = []
    for i in range(ser_of_lists.shape[0]):
        big_list += ser_of_lists.iloc[i]
    most_freq_word_keys_counts = Counter(big_list).most_common(cfg.n_most_frequent)
    most_freq_word_keys = [most_freq_word_keys_counts[i][0] for i in range(len(most_freq_word_keys_counts))]
    return most_freq_word_keys


def least_common_values(array, to_find=None):
    counter = Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))



def build_tf_idf_value(k,v, idf_dict):
    if k in idf_dict.keys():
        tf_idf = v*idf_dict[k]
    else:
        tf_idf = 0
    return tf_idf

def get_lemma(word):
    return WordNetLemmatizer().lemmatize(word)

def build_lemmatized_lists(n_list_ser):
    letters_lemmatized_ser = n_list_ser.apply(lambda x: lemmatize_list(x))
    return letters_lemmatized_ser

def lemmatize_list(n_list):
    return [get_lemma(x) for x in n_list]

def calc_tf(lemmatized_df):
    lemmatized_df['dict_count'] = lemmatized_df.apply(lambda row: dict(Counter(row.lemma_list)),axis=1)
    lemmatized_df['tf'] = lemmatized_df.apply(lambda row: {k:v/row.total_word_count for k,v in
                                                                         row.dict_count.iteritems()})
