# Установка и запуск в докере

Сохраните в корневой директори .env с переменными окружения для подключения к PG (пример в /config/.env.example)

Инициализация таблиц, индексов и функций БД: /init.sql (PG выполнит скрипт при первом запуске контейнера)

## Сборка докера
```
docker build -t postrges_db_reviews .
```

## Запуск докера
```
docker run --name db_reviews \
  -e POSTGRES_PASSWORD=password \
  -p 5435:5432 \
  -d postrges_db_reviews
```

# Загрузка данных и процессинг

## Загрузка
```
python -m python.data_loader # загружаем данные и подключаем триггер к таблице reviews для обработки новых отзывов
```

## NLP-обработка (тональность и ключевые фразы)
```
python -m python.nlp_processor # пакетная обработка необратанных отзывов (batch_size = 100)
```

## Генерация эмбеддингов
```
python -m python.embedding_generator # пакетная генерация эмбеддингов (batch_size = 10)
```

# Проверка работы триггера

## Добавляем новый отзыв
```
INSERT INTO reviews (
    review_id,
    product_id,
    user_id,
    review_text,
    rating
)
VALUES (
    1000,
    1000,
    1000,
    'Отличный товар, спасибо продавцу!',
    5
);
```

## Проверяем сгенерированные данные (ключевые фразы, тональность, эмбеддинги)
```
SELECT * FROM key_phrases WHERE review_id = 1000;
SELECT * FROM sentiment_analysis WHERE review_id = 1000;
SELECT * FROM review_embeddings WHERE review_id = 1000;
```
