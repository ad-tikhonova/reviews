from psycopg2.extras import execute_batch
from tqdm import tqdm

from python.db_connection import DatabaseConnection


class NLPProcessor:
    def process_reviews_batch(self, batch_size: int = 100):
        """Обработка отзывов батчами (тональность и ключевые фразу) и сохранение результатов в БД"""
        with DatabaseConnection() as conn:
            with conn.cursor() as cursor:
                # Получаем количество необработанных отзывов
                cursor.execute("""
                    SELECT COUNT(*) FROM reviews r
                    LEFT JOIN sentiment_analysis sa ON r.review_id = sa.review_id
                    WHERE sa.review_id IS NULL
                """)
                total_reviews = cursor.fetchone()[0]

                if total_reviews == 0:
                    print("Все отзывы уже обработаны")
                    return

                # Обрабатываем отзывы батчами
                for offset in tqdm(
                    range(0, total_reviews, batch_size), desc="Обработка отзывов"
                ):
                    cursor.execute(
                        """
                            SELECT r.review_id, r.review_text
                            FROM reviews r
                            LEFT JOIN sentiment_analysis sa ON r.review_id = sa.review_id
                            WHERE sa.review_id IS NULL
                            ORDER BY r.created_at DESC
                            LIMIT %s OFFSET %s
                        """,
                        (batch_size, offset),
                    )

                    batch = cursor.fetchall()

                    sentiment_data = []
                    key_phrases_data = []

                    for review in batch:
                        review_id, review_text = review

                        # извлекаем ключевые фразы
                        cursor.execute(
                            "SELECT * FROM extract_key_phrases(%s)",
                            (review_text,),
                        )
                        result = cursor.fetchone()
                        key_phrases_data.append((review_id, result[0]))

                        # извлекаем тональность
                        cursor.execute(
                            "SELECT * FROM analyze_sentiment_transformers(%s)",
                            (review_text,),
                        )
                        analyze_sentiment = cursor.fetchone()
                        sentiment_score = analyze_sentiment[0]
                        sentiment_label = analyze_sentiment[1]

                        sentiment_data.append(
                            (review_id, sentiment_score, sentiment_label)
                        )

                        # Сохраняем ключевые фразы
                        execute_batch(
                            cursor,
                            """
                            INSERT INTO key_phrases (
                                review_id, phrases
                            ) VALUES (%s, %s)
                            ON CONFLICT (review_id) DO NOTHING
                        """,
                            key_phrases_data,
                        )

                        # Сохраняем тональность
                        execute_batch(
                            cursor,
                            """
                            INSERT INTO sentiment_analysis (
                                review_id, sentiment_score, sentiment_label
                            ) VALUES (%s, %s, %s)
                            ON CONFLICT (review_id) DO NOTHING
                        """,
                            sentiment_data,
                        )

                        conn.commit()


if __name__ == "__main__":
    processor = NLPProcessor()
    processor.process_reviews_batch()