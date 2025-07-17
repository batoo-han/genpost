import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from currentsapi import CurrentsAPI

load_dotenv()

# Логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

openai_api_key = os.getenv("OPENAI_API_KEY")
currentsapi_key = os.getenv("CURRENTS_API_KEY")

if not openai_api_key or not currentsapi_key:
    logging.error("Не заданы ключи OPENAI_API_KEY/CURRENTS_API_KEY!")
    raise ValueError("Установите ключи OPENAI_API_KEY и CURRENTS_API_KEY")

client = OpenAI(api_key=openai_api_key)
news_client = CurrentsAPI(api_key=currentsapi_key)


class Topic(BaseModel):
    topic: str


def get_recent_news(topic: str):
    logging.info(f"Запрос новостей через currentsapi-python по теме: {topic}")
    try:
        news_data = news_client.latest_news(language='ru', keywords=topic)
        news_list = news_data.get("news", [])
        if not news_list:
            logging.warning("Свежих новостей не найдено через CurrentsAPI!")
            return "Свежих новостей не найдено."
        titles = [article.get("title", "") for article in news_list[:5]]
        logging.info(f"Получено {len(titles)} новостей: {titles}")
        return "\n".join(titles)
    except Exception as e:
        logging.error(f"Ошибка CurrentsAPI: {e}")
        return "Свежих новостей не найдено: ошибка CurrentsAPI."


def generate_content(topic: str):
    logging.info(f"Генерация контента по теме: {topic}")
    recent_news = get_recent_news(topic)
    try:
        # Генерация заголовка
        title_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Придумайте привлекательный и точный заголовок для статьи на тему '{topic}', с учётом актуальных новостей:\n{recent_news}. Заголовок должен быть интересным и ясно передавать суть темы."
            }],
            max_tokens=60,
            temperature=0.5,
            stop=["\n"]
        )
        title = title_resp.choices[0].message.content.strip()

        # Генерация мета-описания
        meta_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Напишите мета-описание для статьи с заголовком: '{title}'. Оно должно быть полным, информативным и содержать основные ключевые слова."
            }],
            max_tokens=120,
            temperature=0.5,
            stop=["."]
        )
        meta_description = meta_resp.choices[0].message.content.strip()

        # Генерация полного текста статьи
        content_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Ты профессиональный журналист, который пишет глубокии и интересные статьи.
                Напишите подробную статью на тему '{topic}', используя последние новости:\n{recent_news}. 
                Статья должна быть:
                1. Информативной и логичной
                2. Содержать не менее 2000 символов
                3. Иметь четкую структуру с подзаголовками
                4. Включать анализ текущих трендов
                5. Иметь вступление, основную часть и заключение
                6. Включать примеры из актуальных новостей
                7. Каждый абзац должен быть не менее 3-4 предложений
                8. Текст должен быть легким для восприятия и содержательным"""
            }],
            max_tokens=2000,
            temperature=0.3,
            presence_penalty=0.6,
            frequency_penalty=0.6
        )
        post_content = content_resp.choices[0].message.content.strip()

        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }

    except Exception as e:
        logging.error(f"Ошибка генерации контента: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации контента: {str(e)}")


@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    return generate_content(topic.topic)


@app.get("/")
async def root():
    return {"message": "Service is running"}


@app.get("/heartbeat")
async def heartbeat_api():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn

    # Запуск приложения с указанием порта
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
