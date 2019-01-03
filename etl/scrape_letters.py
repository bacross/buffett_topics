import cfg
import requests
from bs4 import BeautifulSoup
from fastparquet import write
import pandas as pd
from time import sleep
from random import randint

def get_text_from_letter(n_letter_url):
    if n_letter_url.endswith('.html'):
        txt = get_text_from_letter_html(n_letter_url)
    else:
        #ToDo: add pdf scraping function
        txt=''
        print('pdf func not built yet')
    return txt

def get_text_from_letter_html(n_url):
    soup = BeautifulSoup(open(n_url), 'html.parser')
    txt = soup.get_text()
    return txt

def clean_soup_text(soup_text, n_letter_url):
    if cfg.start_phrase_a in soup_text:
        txt = soup_text.split(cfg.start_phrase_a,1)[1]
    else:
        if cfg.start_phrase_b in soup_text:
            txt = soup_text.split(cfg.start_phrase_b, 1)[1]
        else:
            txt = soup_text.split(cfg.start_phrase_c, 1)[1]
    txt = txt.replace('\n','').replace('\r','')
    print(n_letter_url[-9:-5])
    return txt

def scrape_letters(list_of_letter_urls):
    corp_list = []
    year_list = []
    for letter_url in list_of_letter_urls:
        #sleep(randint(1,4))
        txt = get_text_from_letter(letter_url)
        txt = clean_soup_text(txt,letter_url)
        corp_list.append(txt)
        year_list.append(int(letter_url[-9:-5]))
    txt_df = pd.DataFrame(corp_list)
    txt_df.columns = ['letter_text']
    txt_df.index = year_list
    txt_df.to_parquet(cfg.letters_df_parq_path,compression='gzip')
    return txt_df


