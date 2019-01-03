# define paths
repo_path = '.'
data_path = repo_path+'/data'
letters_path = data_path+'/html_letters'
letters_df_parq_path = data_path+'/letters_df.parq'
proc_letters_df_parq_path = data_path+'/proc_letters_df.parq'
fin_word_corpus = data_path+'/LoughranMcDonald_MasterDictionary_2016.csv'
tf_idf_df_path = data_path+'/tf_idf_df.parq'
lsi_topic_path = data_path+'/lsi_topics.parq'
lda_topic_path = data_path+'/lda_topics.parq'

# proc flags
just_run_lda_flag = False
refresh_letter_scrape_flag = False
proc_letters_refresh_flag = False
tf_idf_refresh_flag = True
lsi_model_refresh_flag = True
gensim_lda_model_refresh_flag = True

# url info
shareholder_letters_url = 'http://www.berkshirehathaway.com/letters'

years = range(1977, 2002, 1)

letter_url_list = ["Chairman's Letter - "+str(yr)+".html" for yr in years]

letter_path_list = [letters_path+'/'+letter for letter in letter_url_list]

# string info
start_phrase_a = 'To the Shareholders of Berkshire Hathaway Inc.:'
start_phrase_b = 'To the Stockholders of Berkshire Hathaway Inc.:'
start_phrase_c = 'To the Shareholders of Berkshire Hathaway Inc. :'

# prevent conversion of the word 'null' to NaN
na_list = ['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NaN', 'n/a', 'nan']

# n most frequent words across all documents in order to reduce dimensionality of tf_idf
n_most_frequent = 1000

# modelling config
n_topics = 5
n_words_to_describe_topics = 10
rem_list = ['berkshire', 'hathaway', 'berkshire_hathaway', 'warren', 'charlie', 'omaha', 'le','than_zero',
            'general_re','ralph','salomon','chuck','frank','brown','fechheimer','coke','wells','fargo','wells_fargo',
            'american_express','american','express','jack','gene','wppss','disney','lou','rockford','helzberg','illinois',
            'ike','profit_le','kirby','cap_city','friedman','rjr','dave','arcata','dave','usair','well_fargo','abc',
            'peter','heldmans','blumkins','willey','safeco','shey','kansa','mike','wesco_financial','wesco','eja',
            'gorat','tony','dexter','shaw','aksarben','netjets','phil', 'stan','redwood','midamerican','super_cat',
            'chip','orpheum','thousand','million','billion','hundred','though','would','insurance','reinsurance']

# Set LDA training parameters.
num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = None