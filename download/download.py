import asyncio
from pyrogram import Client
import pandas as pd
import json

LIMIT = 10000


async def main():

    with open('credentials.txt', 'r') as f:
        username, phone_number, api_id, api_hash = f.read().strip().split('\n')

    client = Client(username, phone_number=phone_number, api_id=api_id, api_hash=api_hash)
    await client.start()

    channels = []
    with open('channels.json', 'r') as f:
        data = json.load(f)
        for category in data:
            channels += [(x, category) for x in data[category]]

    for channel, category in channels:
        print(f'Downloading posts from {channel}')
        df = pd.DataFrame(columns=['time', 'text', 'channel_id'])

        texts = []
        dates = []
        ids = []

        index = 0
        async for message in client.get_chat_history(channel):
            txt = message.text
            if txt is None:
                txt = message.caption
            if txt:
                txt = txt.replace('\n', ' ')
            date = message.date
            channel_id = message.sender_chat.id
            texts.append(txt)
            dates.append(date)
            ids.append(channel_id)
            index += 1
            if index == LIMIT:
                break

        df['time'] = dates
        df['text'] = texts
        df['channel_id'] = ids
        # df.loc[:, 'expected_cat'] = category

        df.to_csv(f'dummy/{channel[1:]}.csv', index=False)

    await client.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
