import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

load_dotenv(override=True)


class DatabaseConnection:
    def __init__(self):
        self.connection = None

    def __enter__(self):
        """Устанавливаем соединение"""
        try:
            self.connection = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                cursor_factory=DictCursor,
            )
            return self.connection
        except psycopg2.Error as e:
            print(f"Ошибка подключения к базе данных: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрываем соединение"""
        if self.connection:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
            self.connection.close()


def test_connection():
    """Тестовая функция для проверки подключения"""
    try:
        with DatabaseConnection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                db_version = cursor.fetchone()
                print(f"Подключение успешно. Версия PostgreSQL: {db_version[0]}")
        return True
    except Exception as e:
        print(f"Ошибка при тестировании подключения: {e}")
        return False


if __name__ == "__main__":
    test_connection()