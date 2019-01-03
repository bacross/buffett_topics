import cfg
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def lsi_model_topics(tf_idf_df):

    svd_model = TruncatedSVD(n_components=cfg.n_topics, algorithm='randomized', n_iter=100, random_state=122)

    svd_model.fit(tf_idf_df)

    terms = tf_idf_df.columns

    topic_dict = {}
    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:cfg.n_words_to_describe_topics]
        topic_dict['Topic %s' %(i)] = [t[0] for t in sorted_terms]

    topics = pd.DataFrame.from_dict(topic_dict, orient='index')
    topics.columns = ['word_%s' %(i) for i in range(topics.shape[1])]
    topics.to_parquet(cfg.lsi_topic_path,compression='gzip')

    return topics
