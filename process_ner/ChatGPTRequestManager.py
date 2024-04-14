import asyncio
import traceback
import time

import openai
from openai import OpenAI

SYSTEM_PROMPT = """Ты профессиональный финансовый аналитик. Тебе необходимо оценить Sentiment score сообщения по каждой компании числами от 1 до 5, где
1: Очень негативное относительно компании или дана рекомендация "продавать"
2: Скорее негативная новость, но нет явной рекомендации "продавать"
3: Нейтральная новость. Важно! Если текст "Акция выросла/упала на 40% за день", то считается нейтральной новостью; но "Акция выросла на 40% за день, потому что ..."
Считается положительной новостью (sentiment_score > 3), т.к. есть объяснение.
4: Что-то положительное про компанию (но нет рекомендации "покупать")
5: Очень положительное или есть рекомендация "покупать" или "входит в подборку наших супер-акций"

Например 
"YNDX сегодня упал на 5%" - 3, нейтральная новость
"YNDX сегодня упал на 5%, потому что компания не выполнила план по прибыли" - 2, скорее негативная новость
"YNDX признан банкротом" - 1, очень негативная новость

Тебе будет дан текст сообщения и список компаний. Выведи их оценки в формате: "имя компании: оценка". Выведи ровно столкьо строк, сколько компаний в списке.
Не забывай что просто новость о том что цена какой-то акции выросла/упала является нейтральной новостью (оценка 3).
"""


class RequestManager:
    def __init__(self, model: str = 'gpt-3.5-turbo'):
        with open('../openai_key.txt', 'r') as f:
            self.client = OpenAI(
                api_key=f.read().strip()
            )

    def write_one_message_with_role(self, message: str,
                                    system_role: str = SYSTEM_PROMPT,
                                    model: str = 'gpt-3.5-turbo'
                                    ):

        while True:
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": message},
                    ],
                    model=model,
                )
                break
            except Exception as e:  # openai.error.RateLimitError:
                print(f"Failed: {e}")
                time.sleep(10)
        return response.choices[0].message.content


def main():
    manager = RequestManager()
    result = manager.write_one_message_with_role(
        '{$ABIO} пробьет 111, дальше к 105 пойдет, у сбера так себе отчет вышел, Яндекс подрос на 5% \n'
        'Компании: $ABIO, Сбербанк, YNDX'
    )
    print(result)


if __name__ == "__main__":
    main()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
