from psycopg2.extras import execute_batch
from tqdm import tqdm

from python.db_connection import DatabaseConnection


class EmbeddingGenerator:
    def process_reviews_batch(self, batch_size: int = 10):
        """Обработка отзывов батчами и сохранение эмбеддингов в БД"""
        with DatabaseConnection() as conn:
            with conn.cursor() as cursor:
                # Получаем количество необработанных отзывов
                cursor.execute("""
                    SELECT COUNT(*) FROM reviews r
                    LEFT JOIN review_embeddings re ON r.review_id = re.review_id
                    WHERE re.review_id IS NULL
                """)
                total_reviews = cursor.fetchone()[0]

                if total_reviews == 0:
                    print("Все отзывы уже обработаны")
                    return

                # Обрабатываем отзывы батчами
                for offset in tqdm(
                    range(0, total_reviews, batch_size), desc="Генерация эмбеддингов"
                ):
                    cursor.execute(
                        """
                            SELECT r.review_id, r.review_text
                            FROM reviews r
                            LEFT JOIN review_embeddings re ON r.review_id = re.review_id
                            WHERE re.review_id IS NULL
                            ORDER BY r.created_at DESC
                            LIMIT %s OFFSET %s
                        """,
                        (batch_size, offset),
                    )

                    batch = cursor.fetchall()
                    embeddings_data = []

                    for review in batch:
                        review_id, review_text = review

                        # извлекаем ключевые фразы
                        cursor.execute(
                            "SELECT * FROM generate_text_embeddings(%s)",
                            (review_text,),
                        )
                        result = cursor.fetchone()

                        embeddings_data.append(
                            (
                                review_id,
                                result[0],
                            )
                        )

                    # Сохраняем эмбеддинги в базу данных
                    execute_batch(
                        cursor,
                        """
                            INSERT INTO review_embeddings (
                                review_id, embedding_vector
                            ) VALUES (%s, %s)
                            ON CONFLICT (review_id) DO NOTHING
                        """,
                        embeddings_data,
                    )

                    conn.commit()


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.process_reviews_batch()