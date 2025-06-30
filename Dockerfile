FROM postgres:17-bullseye

# создаем виртуальное окружение и устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y \
    postgresql-plpython3-17 \
    python3.9 \
    python3.9-venv \
    postgresql-17-pgvector \
    && python3.9 -m venv /opt/pg_python \
    && /opt/pg_python/bin/pip install --upgrade pip \
    && /opt/pg_python/bin/pip install transformers torch nltk sentence-transformers \
    && chown -R postgres:postgres /opt/pg_python

# инициализация таблиц, индексов и функций БД
COPY init.sql /docker-entrypoint-initdb.d/