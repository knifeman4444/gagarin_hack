import pandas as pd
import os


sus = []

directory = 'data_large'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    df = pd.read_csv(f)
    print(df)
    df = df.drop('channel_id', axis=1)
    sus.append(df)

directory = 'data_pulse'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    sus.append(pd.read_csv(f))

df = pd.concat(sus)

df.to_csv(f'all_data.csv', index=False)
