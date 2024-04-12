import pickle
import pandas as pd
import re

path = 'data/sentiment_dataset'

with open(f'{path}/mentions texts.pickle', 'rb') as file:
    mentions_texts = pickle.load(file)

with open(f'{path}/sentiment_texts.pickle', 'rb') as file:
    sent_texts = pickle.load(file)

mentions = pd.read_csv(f'{path}/mentions.csv', index_col=0)
sent = pd.read_csv(f'{path}/sentiment.csv', index_col=0)

issuers = pd.read_excel(f'{path}/issuers.xlsx', index_col=0)
synonyms = pd.read_excel(f'{path}/names and synonyms.xlsx')

synonyms_dict = {}
for i in range(synonyms.shape[0]):

    idx = synonyms.issuerid.iloc[i]
    synonyms_dict[idx] = synonyms.iloc[i].dropna().values[1:]
    synonyms_dict[idx] = [x.strip().lower() for x in synonyms_dict[idx]]

    # Remove rx followed by space or with space before ("rx " or " rx")
    synonyms_dict[idx] = [re.sub(r'rx\s', '', x) for x in synonyms_dict[idx]]
    synonyms_dict[idx] = [re.sub(r'\srx', '', x) for x in synonyms_dict[idx]]

mentions_texts['MessageID'] = mentions_texts.ChannelID.astype(str) + mentions_texts.messageid.astype(str)
mentions_texts.drop(columns=['ChannelID', 'messageid'], inplace=True)
all_mentions = mentions_texts.groupby('MessageID').issuerid.apply(list)
texts = mentions_texts[['MessageID', 'MessageText']].drop_duplicates()

