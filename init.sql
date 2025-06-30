-- Создаем расширения при инициализации БД
CREATE EXTENSION IF NOT EXISTS plpython3u;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Добавляем все таблицы и функции

-- ТАБЛИЦЫ

-- Информация о товарах.
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Данные пользователей (авторов отзывов).
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    user_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Отзывы с привязкой к товарам и пользователям.
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products (product_id),
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    review_text TEXT NOT NULL,
    rating FLOAT CHECK (rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Результаты анализа тональности (оценка и метка)
CREATE TABLE sentiment_analysis (
    review_id BIGINT UNIQUE,
    sentiment_score FLOAT NOT NULL CHECK (sentiment_score BETWEEN -1 AND 1),
    sentiment_label VARCHAR(20) NOT NULL CHECK (
        sentiment_label IN ('positive', 'neutral', 'negative')
    ),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
);

-- Извлеченные ключевые фразы
CREATE TABLE key_phrases (
    review_id BIGINT UNIQUE,
    phrases TEXT[] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (review_id, phrases),
    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
);

-- Таблица для хранения векторных представлений текста
CREATE TABLE review_embeddings (
    review_id BIGINT UNIQUE,
    embedding_vector VECTOR(384) NOT NULL,
    model_version VARCHAR(50) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
);

-- Индексы для векторного поиска.
CREATE INDEX idx_embedding_vector ON review_embeddings USING ivfflat (embedding_vector vector_l2_ops)
WITH
    (lists = 100);

-- Поле для хранения векторы в таблице reviews
ALTER TABLE reviews ADD COLUMN ts_vector tsvector;

-- GIN-индекс для ускорения поиска
CREATE INDEX idx_reviews_tsvector ON reviews USING gin(ts_vector);

-- Функции для обновления ts_vector
CREATE OR REPLACE FUNCTION update_review_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_vector := 
        setweight(to_tsvector('russian', COALESCE(NEW.review_text, '')), 'A');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Триггер для автоматического обновления ts_vector
CREATE TRIGGER trigger_update_review_tsvector
BEFORE INSERT OR UPDATE OF review_text ON reviews
FOR EACH ROW
EXECUTE FUNCTION update_review_tsvector();

-- Генерация эмбедингов
CREATE OR REPLACE FUNCTION generate_text_embeddings(text TEXT, model_name TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2')
RETURNS FLOAT[] AS $$
import sys
sys.path.insert(0, '/opt/pg_python/lib/python3.9/site-packages')

if 'embedding_model' not in GD:
    try:
        from sentence_transformers import SentenceTransformer
        GD['embedding_model'] = SentenceTransformer(model_name)
    except:
        plpy.error("Требуется установка sentence-transformers. Выполните: pip install sentence-transformers torch")

try:
    # Генерация эмбеддинга и преобразование в список
    embedding = GD['embedding_model'].encode(text)
    return [float(x) for x in embedding]
except Exception as e:
    plpy.error(f"Ошибка генерации эмбеддинга: {str(e)}")
$$ LANGUAGE plpython3u;

-- Функция извлеченияя ключевых фраз
CREATE OR REPLACE FUNCTION extract_key_phrases(
    text TEXT,
    max_phrases INTEGER DEFAULT 5
)
RETURNS text[] AS $$
import sys
# Добавляем путь к виртуальному окружению
sys.path.insert(0, '/opt/pg_python/lib/python3.9/site-packages')

import subprocess
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.data.path.append('/var/lib/postgresql/nltk_data')
nltk.download('stopwords', download_dir='/var/lib/postgresql/nltk_data')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))

# Функция для очистки и токенизации текста
def preprocess_text(text):
    # Удаляем спецсимволы и цифры
    text = re.sub(r'[^а-яА-Яa-zA-Z\s]', '', text.lower())
    # Токенизируем
    tokens = text.split()
    # Удаляем стоп-слова и короткие слова
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

# Обрабатываем входной текст
tokens = preprocess_text(text)
if not tokens:
    return []

# Создаем документ для TF-IDF
doc = [' '.join(tokens)]

# Создаем TF-IDF векторайзер
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=max_phrases * 2,
    stop_words=list(stop_words)
)

try:
    # Применяем TF-IDF
    X = vectorizer.fit_transform(doc)
    
    # Получаем фичи и их веса
    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    
    # Сортируем по убыванию релевантности
    phrases_with_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ фраз
    return [phrase for phrase, score in phrases_with_scores[:max_phrases]]
        
except ValueError as e:
    plpy.notice(f"Ошибка TF-IDF: {str(e)}")
    return []
$$ LANGUAGE plpython3u;


-- Анализ тональности
CREATE OR REPLACE FUNCTION analyze_sentiment_transformers(
    review_text TEXT,
    model_name TEXT DEFAULT 'blanchefort/rubert-base-cased-sentiment'
) 
RETURNS TABLE(sentiment_score FLOAT, sentiment_label TEXT) AS $$
import sys
# Добавляем путь к виртуальному окружению
sys.path.insert(0, '/opt/pg_python/lib/python3.9/site-packages')

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Кэш для хранения модели и токенизатора
model_cache = {}

if 'sentiment_model' not in GD:
    try:
        # Загружаем модель и токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Используем GPU если доступен
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            model = model.cuda()
        
        # Создаем pipeline для анализа тональности
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        GD['sentiment_model'] = sentiment_analyzer
        GD['model_name'] = model_name
    except Exception as e:
        plpy.error(f"Ошибка загрузки модели: {str(e)}")

try:
    # Анализируем текст
    result = GD['sentiment_model'](review_text)[0]
    
    # Преобразуем результат в нужный формат
    label = result['label']
    score = result['score']
    
    # Нормализуем оценку в диапазон -1..1
    if label == 'POSITIVE':
        sentiment_score = score
    elif label == 'NEGATIVE':
        sentiment_score = -score
    else:
        sentiment_score = 0
    
    # Возвращаем результат
    yield (sentiment_score, label.lower())
except Exception as e:
    plpy.notice(f"Ошибка анализа текста: {str(e)}")
    yield (0.0, 'neutral', 0.0)
$$ LANGUAGE plpython3u;


-- Проверка доступности PL/Python и установленных пакетов в виртульном окружении
CREATE OR REPLACE FUNCTION test_plpython()
RETURNS TEXT AS $$
try:
    import sys
    # Добавляем путь к виртуальному окружению
    sys.path.insert(0, '/opt/pg_python/lib/python3.9/site-packages')
    from transformers import pipeline
    from torch import __version__ as torch_version
    return f"Packages доступны (transformers, torch {torch_version})"
except ImportError as e:
    return f"Ошибка импорта: {str(e)}. Убедитесь, что пакеты установлены в venv!"
$$ LANGUAGE plpython3u;


-- Создаем функцию-обработчик для триггера
CREATE OR REPLACE FUNCTION process_new_review()
RETURNS TRIGGER AS $$
BEGIN
    -- Обновляем ts_vector
    NEW.ts_vector := setweight(to_tsvector('russian', COALESCE(NEW.review_text, '')), 'A');

    -- Анализ тональности
    INSERT INTO sentiment_analysis (review_id, sentiment_score, sentiment_label)
    SELECT 
        NEW.review_id,
        sentiment_score,
        sentiment_label
    FROM analyze_sentiment_transformers(NEW.review_text)
    ON CONFLICT (review_id) DO NOTHING;
    
    -- Извлечение ключевых фраз
    INSERT INTO key_phrases (review_id, phrases)
    VALUES (
        NEW.review_id,
        ARRAY(SELECT extract_key_phrases(NEW.review_text))
    )
    ON CONFLICT (review_id) DO NOTHING;

    -- Генерация эмбеддингов
    INSERT INTO review_embeddings (review_id, embedding_vector)
    VALUES (
        NEW.review_id,
        generate_text_embeddings(NEW.review_text)
    )
    ON CONFLICT (review_id) DO NOTHING;

    -- перерасчет данных в материализованном представлении
    REFRESH MATERIALIZED VIEW product_rating_stats;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- Создания материализованного представленияя
CREATE MATERIALIZED VIEW product_rating_stats AS
SELECT 
    p.product_id,
    p.product_name,
    ROUND(AVG(r.rating)::numeric, 2) as avg_rating,
    COUNT(r.review_id) as review_count,
    ROUND(STDDEV(r.rating)::numeric, 2) as rating_stddev,
    MIN(r.rating) as min_rating,
    MAX(r.rating) as max_rating,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.rating) as median_rating
FROM 
    products p
    LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY 
    p.product_id, p.product_name;