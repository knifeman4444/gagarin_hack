import pickle
import pandas as pd
from .utils import normalize_text
import re
import os

path = 'data/sentiment_dataset'
path = os.environ.get('HACK_DATA_PATH', path)

with open(f'{path}/mentions texts.pickle', 'rb') as file:
    mentions_texts = pickle.load(file)

with open(f'{path}/sentiment_texts.pickle', 'rb') as file:
    sent_texts = pickle.load(file)

issuers = pd.read_excel(f'{path}/issuers.xlsx', index_col=0)
synonyms = pd.read_excel(f'{path}/names and synonyms.xlsx')

synonyms_dict = {}
tickers = {}

for i in range(synonyms.shape[0]):

    idx = synonyms.issuerid.iloc[i]
    name = synonyms.iloc[i, 1]
    ticker_rx = synonyms.iloc[i, 3]
    ticker = synonyms.iloc[i, 4]

    if not pd.isna(ticker_rx):
        ticker_rx = ticker_rx.lower().replace(' rx', '').replace(' li', '').strip()
    if not pd.isna(ticker):
        ticker = ticker.lower().strip()
    if not pd.isna(ticker) and not re.match(r'^[a-zA-Z]+$', ticker):
        ticker = ticker_rx
    tickers[idx] = ticker_rx if (pd.isna(ticker) or ticker == '') else ticker

    synonyms_dict[idx] = synonyms.iloc[i, 5:].dropna().values.tolist() + ([name] if len(name) <= 20 else [])
    synonyms_dict[idx] = [normalize_text(x) for x in synonyms_dict[idx]]


mentions_texts['MessageID'] = mentions_texts.ChannelID.astype(str) + mentions_texts.messageid.astype(str)
mentions_texts.drop(columns=['ChannelID', 'messageid'], inplace=True)
all_mentions = mentions_texts.groupby('MessageID').issuerid.apply(list)
texts = mentions_texts[['MessageID', 'MessageText']].drop_duplicates()


def get_name_by_id(issuer_id: int) -> str:
    row = synonyms[synonyms.issuerid == issuer_id]
    if row.shape[0] == 0:
        raise ValueError(f'No issuer with id {issuer_id}')
    if row.shape[0] > 1:
        raise ValueError(f'Multiple issuers with id {issuer_id}?')
    return row.iloc[0, 1]

