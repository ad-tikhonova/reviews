import os
import csv
from psycopg2 import sql
from psycopg2.extras import execute_batch
from tqdm import tqdm

from python.db_connection import DatabaseConnection


class DataLoader:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.batch_size = 1000  # Размер батча для вставки

    def _get_user_id(self, cursor, user_name):
        """Получаем или создаем пользователя и возвращаем его ID"""
        query = sql.SQL("""
            INSERT INTO users (user_name)
            VALUES (%s)
            ON CONFLICT (user_name) DO UPDATE SET
                user_name = EXCLUDED.user_name
            RETURNING user_id;
        """)
        cursor.execute(query, (user_name,))
        return cursor.fetchone()[0]

    def _get_product_id(self, cursor, product_name):
        """Получаем или создаем продукт и возвращаем его ID"""
        query = sql.SQL("""
            INSERT INTO products (product_name)
            VALUES (%s)
            ON CONFLICT (product_name) DO UPDATE SET
                product_name = EXCLUDED.product_name
            RETURNING product_id;
        """)
        cursor.execute(query, (product_name,))
        return cursor.fetchone()[0]

    def load_data(self):
        """Основной метод загрузки данных из CSV в БД"""
        if not os.path.exists(self.csv_file_path):
            print(f"Файл {self.csv_file_path} не найден!")
            return False

        try:
            with (
                DatabaseConnection() as conn,
                open(self.csv_file_path, "r", encoding="utf-8") as csv_file,
            ):
                csv_reader = csv.DictReader(csv_file)
                total_rows = sum(1 for _ in csv_reader)
                csv_file.seek(0)  # Возвращаемся в начало файла
                next(csv_reader)  # Пропускаем заголовок

                with conn.cursor() as cursor:
                    # Подготовка батча для вставки отзывов
                    batch = []
                    insert_query = sql.SQL("""
                        INSERT INTO reviews (
                            product_id, user_id, 
                            review_text, rating
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (review_id) DO NOTHING;
                    """)

                    for row in tqdm(
                        csv_reader, total=total_rows, desc="Загрузка данных"
                    ):
                        try:
                            # Получаем или создаем пользователя
                            user_id = self._get_user_id(cursor, row["reviewerName"])

                            # Получаем или создаем продукт
                            product_id = self._get_product_id(cursor, row["name"])

                            # Добавляем данные отзыва в батч
                            batch.append(
                                (
                                    product_id,
                                    user_id,
                                    row["text"],
                                    float(row["mark"]),
                                )
                            )

                            # Когда батч заполнен, записываем данные
                            if len(batch) >= self.batch_size:
                                execute_batch(cursor, insert_query, batch)
                                batch = []

                        except (ValueError, KeyError) as e:
                            print(f"Ошибка обработки строки: {row}. Ошибка: {e}")
                            continue

                    # если батч не пустой, записываем данные
                    if len(batch) > 0:
                        execute_batch(cursor, insert_query, batch)

                    # Создаем триггер для таблицы reviews (подключаем тут, чтобы во время загрузки всех данные не нагрузить PG)
                    cursor.execute(
                        """
                        CREATE TRIGGER trigger_process_review
                        AFTER INSERT ON reviews
                        FOR EACH ROW
                        EXECUTE FUNCTION process_new_review();
                        """
                    )

                    # перерасчет данных в материализованном представлении
                    cursor.execute(
                        """
                        REFRESH MATERIALIZED VIEW product_rating_stats;
                        """
                    )

                print("Загрузка данных успешно завершена!")
                return True

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return False


if __name__ == "__main__":
    loader = DataLoader("sample_data.csv")
    loader.load_data()