import logging
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords

from python.db_connection import DatabaseConnection

nltk.download("stopwords")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewSimilarityAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=stopwords.words("russian"),
        )

    def find_similar_by_embeddings(self, target_review_id, num_results=5):
        """
        Поиск похожих отзывов по векторным представлениям
        """

        with DatabaseConnection() as conn:
            with conn.cursor() as cursor:
                # Получаем эмбеддинг целевого отзыва
                cursor.execute(
                    """
                    SELECT embedding_vector 
                    FROM review_embeddings 
                    WHERE review_id = %s
                """,
                    (target_review_id,),
                )
                target_embedding = cursor.fetchone()

                if not target_embedding:
                    logger.error(f"Эмбеддинг для отзыва {target_review_id} не найден")
                    return []

                # Преобразуем строку с эмбеддингом в numpy массив
                try:
                    target_vec = np.array(eval(target_embedding[0]))
                except Exception as e:
                    logger.error(f"Ошибка преобразования эмбеддинга: {str(e)}")
                    return []

                # Получаем все эмбеддинги (кроме целевого)
                cursor.execute(
                    """
                    SELECT review_id, embedding_vector 
                    FROM review_embeddings 
                    WHERE review_id != %s
                """,
                    (target_review_id,),
                )
                embeddings = cursor.fetchall()

                if not embeddings:
                    logger.warning("Не найдено других эмбеддингов для сравнения")
                    return []

                # Преобразуем все эмбеддинги в numpy массивы
                other_ids = []
                other_vecs = []
                for review_id, embedding_str in embeddings:
                    try:
                        vec = np.array(eval(embedding_str))
                        other_ids.append(review_id)
                        other_vecs.append(vec)
                    except Exception as e:
                        logger.warning(
                            f"Ошибка преобразования эмбеддинга отзыва {review_id}: {str(e)}"
                        )
                        continue

                if not other_vecs:
                    logger.warning("Нет валидных эмбеддингов для сравнения")
                    return []

                # Вычисляем косинусную близость
                similarities = cosine_similarity([target_vec], other_vecs)[0]

                # Сортируем результаты
                results = sorted(
                    zip(other_ids, similarities), key=lambda x: x[1], reverse=True
                )[:num_results]

                return results


if __name__ == "__main__":
    analyzer = ReviewSimilarityAnalyzer()

    target_id = 15

    print("\nПохожие отзывы по эмбеддингам:")
    embed_similar = analyzer.find_similar_by_embeddings(target_id)
    for review_id, similarity in embed_similar:
        print(f"ID: {review_id}, Сходство: {similarity:.4f}")