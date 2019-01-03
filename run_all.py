import cfg
import timeit
import pandas as pd
from etl.scrape_letters import scrape_letters
from models.process_letters import clean_docs
from models.tf_idf_custom import build_tf_idf
from models.lsi_topic_modelling import lsi_model_topics
from models.lda_topic_modelling import build_lda_topic_model


# scrape buffett's letters
if cfg.refresh_letter_scrape_flag == True:
    letters_df = scrape_letters(cfg.letter_path_list)
else:
    letters_df = pd.read_parquet(cfg.letters_df_parq_path)


# process letters into string formats for analysis
if cfg.proc_letters_refresh_flag == True:
    proc_letters_df = clean_docs(letters_df)
else:
    proc_letters_df = pd.read_parquet(cfg.proc_letters_df_parq_path)

if cfg.just_run_lda_flag == False:
    # build custom tf-idf
    if cfg.tf_idf_refresh_flag == True:
        tf_idf = build_tf_idf(proc_letters_df['clean_doc'])
    else:
        tf_idf = pd.read_parquet(cfg.tf_idf_df_path)


    # LSI topic modeling
    if cfg.lsi_model_refresh_flag == True:
        lsi_topic_df = lsi_model_topics(tf_idf)
    else:
        lsi_topic_df = pd.read_parquet(cfg.lsi_topic_path)

# gensim topic modeling
lda, corpus, dictionary  = build_lda_topic_model(letters_df)

print('models have been built')