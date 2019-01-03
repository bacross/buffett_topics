import cfg
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def build_lda_topic_model(letters_df):
    # split the documents into tokens
    tokenizer = RegexpTokenizer(r'\w+')
    letters_df['tokenized_doc'] = letters_df['letter_text'].apply(lambda x: tokenizer.tokenize(x.lower()))
    letters_df['tokenized_doc'] = letters_df['tokenized_doc'].apply(
        lambda row: [token for token in row if not token.isnumeric()])
    letters_df['tokenized_doc'] = letters_df['tokenized_doc'].apply(
        lambda row: [token for token in row if len(token) > 1])

    # lemmatize the documents
    lemmatizer = WordNetLemmatizer()
    letters_df['lemmatized_doc'] = letters_df['tokenized_doc'].apply(
        lambda row: [lemmatizer.lemmatize(token) for token in row])

    # build bigrams
    bigram = Phrases(letters_df['lemmatized_doc'], min_count=20)
    for idx in range(len(letters_df)):
        for token in bigram[letters_df['lemmatized_doc'].iloc[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                letters_df['lemmatized_doc'].iloc[idx].append(token)

    # remove words common to letters
    letters_df['lemmatized_doc'] = letters_df['lemmatized_doc'].apply(
        lambda row: [token for token in row if token not in cfg.rem_list])

    # remove stop words
    letters_df['lemmatized_doc'] = letters_df['lemmatized_doc'].apply(
        lambda row: [token for token in row if token not in stop_words])

    # remove rare words
    docs = list(letters_df['lemmatized_doc'])
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)


    # vectorize data
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    #print('Number of LDA unique tokens: %d' % len(dictionary))
    #print('Number of LDA documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = cfg.num_topics
    chunksize = cfg.chunksize
    passes = cfg.passes
    iterations = cfg.iterations
    eval_every = cfg.eval_every

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token


    lda = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto',
                   iterations=iterations, num_topics=num_topics,passes=passes, eval_every=eval_every)

    return lda, corpus, dictionary