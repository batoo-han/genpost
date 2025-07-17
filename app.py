import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

# Инициализация клиента OpenAI с ключом из переменных окружения
openai_api_key = os.getenv("OPENAI_API_KEY")
currentsapi_key = os.getenv("CURRENTS_API_KEY")

if not openai_api_key or not currentsapi_key:
    raise ValueError("Переменные окружения OPENAI_API_KEY и CURRENTS_API_KEY должны быть установлены")

client = OpenAI(api_key=openai_api_key)


class Topic(BaseModel):
    topic: str


def get_recent_news(topic: str):
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {
        "language": "ru",
        "keywords": topic,
        "apiKey": currentsapi_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {response.text}")

    news_data = response.json().get("news", [])
    if not news_data:
        return "Свежих новостей не найдено."

    return "\n".join([article["title"] for article in news_data[:5]])


def generate_content(topic: str):
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

        # Генерация полного контента статьи
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
    port = int(os.getenv("PORT", 8081))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
