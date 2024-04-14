import asyncio
import traceback
import time

import openai
from openai import OpenAI

SYSTEM_PROMPT = """Необходимо оценить Sentiment score компании числами от 1 до 5, где
1: Очень негативное относительно компании или дана рекомендация "продавать",
2: Скорее негативное.
3: Нейтральная новость. Важно! Если текст "Акция выросла на 40% за день", то считается нейтральной новостью; но "Акция выросла на 40% за день, потому что ...
Считается положительной новостью (sentiment_score > 3), т.к. есть объяснение.
4: Что-то положительное.
5: Очень положительное или есть рекомендация "покупать" или "входит в подборку наших супер-акций"
"""


class RequestManager:
    def __init__(self, model: str = 'gpt-3.5-turbo'):
        with open('openai_key.txt', 'r') as f:
            self.client = OpenAI(
                api_key=f.read().strip()
            )

    # async def write_one_message(self, message: str):
    #     while True:
    #         try:
    #             response = await openai.ChatCompletion.acreate(
    #                 model=self.model,
    #                 messages=[
    #                     {'role': 'user', 'content': message}
    #                 ]
    #             )
    #             break
    #         except openai.error.RateLimitError:
    #             await asyncio.sleep(20)
    #         except Exception:
    #             print(traceback.format_exc())
    #             await asyncio.sleep(20)
    #     return response.choices[0].message['content']

    def write_one_message_with_role(self, message: str,
                                          system_role: str = SYSTEM_PROMPT):

        while True:
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "system_role": system_role,
                            "content": message,
                        }
                    ],
                    model="gpt-3.5-turbo",
                )
                # response = openai.ChatCompletion.create(
                #     model=self.model,
                #     messages=[
                #         {'role': 'system', 'content': system_role},
                #         {'role': 'user', 'content': message}
                #     ]
                # )
                break
            except Exception as e:  # openai.error.RateLimitError:
                print(f"Failed: {e}")
                time.sleep(10)
        print(response)
        return response.choices[0].message['content']


def main():
    manager = RequestManager()
    result = manager.write_one_message_with_role('{$ABIO} пробьет 111, дальше к 105 пойдет')
    print(result)


if __name__ == "__main__":
    main()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
