from parser import TinkoffPulse
import pandas as pd

pulse = TinkoffPulse()

with open('old_tickers.txt', 'r') as f:
    old_tickers = f.read().split()

with open('tickers.txt', 'r') as f:
    tickers = f.read().split()

for ticker in tickers:
    if ticker in old_tickers:
        continue
    print(f'Downloading posts for {ticker}')
    df = pd.DataFrame(columns=['time', 'text'])
    texts = []
    dates = []
    try:
        posts = pulse.get_posts_by_ticker(ticker)['items']
    except TypeError:
        print('No posts found')
        continue
    print(f'Found {len(posts)} posts')
    for post in posts:
        dt = post['inserted']
        txt = post['content']['text']
        texts.append(txt)
        dates.append(dt)

    df['time'] = dates
    df['text'] = texts

    df.to_csv(f'data_pulse/{ticker}.csv', index=False)
