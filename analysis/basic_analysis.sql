-- Распределение оценок по всем отзывам
SELECT
    rating,
    COUNT(*) as review_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
FROM
    reviews
GROUP BY
    rating
ORDER BY
    rating;

-- Зависимость между длиной отзыва и оценкой
SELECT
    rating,
    AVG(LENGTH (review_text)) as avg_length,
    COUNT(*) as review_count
FROM
    reviews
GROUP BY
    rating
ORDER BY
    rating;

-- Средний рейтинг по товарам
SELECT
    p.product_id,
    p.product_name,
    ROUND(AVG(r.rating)::numeric, 2) as avg_rating,
    COUNT(r.review_id) as review_count,
    ROUND(
        (
            SUM(
                CASE
                    WHEN r.rating = 5 THEN 1
                    ELSE 0
                END
            ) * 100.0 / COUNT(r.review_id)::numeric
        ),
        1
    ) as percent_5_star
FROM
    products p
    LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY
    p.product_id,
    p.product_name
ORDER BY
    avg_rating DESC,
    review_count DESC;


-- Средний рейтинг с анализом тональности
SELECT 
    p.product_id,
    p.product_name,
    ROUND(AVG(r.rating)::numeric, 2) as avg_rating,
    ROUND(AVG(sa.sentiment_score)::numeric, 3) as avg_sentiment,
    CASE 
        WHEN AVG(sa.sentiment_score) > 0.2 THEN 'positive'
        WHEN AVG(sa.sentiment_score) < -0.2 THEN 'negative'
        ELSE 'neutral'
    END as sentiment_category,
    COUNT(*) as review_count
FROM 
    products p
    JOIN reviews r ON p.product_id = r.product_id
    JOIN sentiment_analysis sa ON r.review_id = sa.review_id
GROUP BY 
    p.product_id, p.product_name
HAVING 
    COUNT(*) >= 5
ORDER BY 
    avg_sentiment DESC;

-- Средняя тональность отзывов по продуктам
SELECT 
    p.product_id,
    p.product_name,
    AVG(sa.sentiment_score) as avg_sentiment,
    COUNT(*) as review_count,
    SUM(CASE WHEN sa.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
    SUM(CASE WHEN sa.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count
FROM 
    products p
    JOIN reviews r ON p.product_id = r.product_id
    JOIN sentiment_analysis sa ON r.review_id = sa.review_id
GROUP BY p.product_id, p.product_name
ORDER BY avg_sentiment DESC
LIMIT 10;

-- Отзывы с расхождением между оценкой и тональностью (аномальные)
SELECT 
    r.review_id,
    r.product_id,
    r.rating,
    sa.sentiment_score,
    sa.sentiment_label,
    r.review_text
FROM 
    reviews r
    JOIN sentiment_analysis sa ON r.review_id = sa.review_id
WHERE 
    (r.rating <= 2 AND sa.sentiment_label = 'positive') OR
    (r.rating >= 4 AND sa.sentiment_label = 'negative')
ORDER BY ABS(r.rating - CASE sa.sentiment_label 
                         WHEN 'positive' THEN 5 
                         WHEN 'negative' THEN 1 
                         ELSE 3 END) DESC
LIMIT 20;

-- Материализованное представление product_rating_stat
-- топ-10 товаров по рейтингу
SELECT 
    product_id,
    product_name,
    avg_rating,
    review_count
FROM 
    product_rating_stats
WHERE 
    review_count >= 5  -- Только товары с достаточным количеством отзывов
ORDER BY 
    avg_rating DESC, 
    review_count DESC
LIMIT 10;

-- Сравнение среднего и медианного рейтинга
SELECT 
    product_id,
    product_name,
    avg_rating,
    median_rating,
    (avg_rating - median_rating) as diff
FROM 
    product_rating_stats
WHERE 
    review_count >= 5
ORDER BY 
    ABS(avg_rating - median_rating) DESC;


-- Полнотекстовый поиск: простой
SELECT 
    review_id,
    product_id,
    review_text,
    ts_headline('russian', review_text, query) as highlight
FROM 
    reviews, 
    plainto_tsquery('russian', 'рубашка') query
WHERE 
    ts_vector @@ query
LIMIT 10;


-- Полнотекстовый поиск: с рейтингом
SELECT 
    r.review_id,
    r.product_id,
    p.product_name,
    r.rating,
    ts_rank_cd(r.ts_vector, query) as rank_score,
    ts_headline('russian', r.review_text, query) as highlight
FROM 
    reviews r
    JOIN products p ON r.product_id = p.product_id,
    plainto_tsquery('russian', 'хороший | отличный') query
WHERE 
    r.ts_vector @@ query
ORDER BY 
    r.rating DESC, 
    rank_score DESC
LIMIT 10;