-- Temporary table for analytics
WITH sales_enriched AS (
  SELECT
    o.order_id,
    DATE(o.order_date) AS order_date,
    FORMAT_DATE('%Y-%m', DATE(o.order_date)) AS order_month,
    p.country,
    p.category,
    SAFE_MULTIPLY(CAST(o.quantity AS FLOAT64), CAST(o.unit_price AS FLOAT64)) AS revenue
  FROM `main-project-712026.test.orders` o
  JOIN `main-project-712026.test.products_customers` p
    ON o.product_id = p.product_id
   AND o.customer_id = p.customer_id
),

sales_summary AS (
  SELECT
    order_month,
    country,
    category,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(revenue) AS total_revenue,
    ROUND(AVG(revenue), 2) AS avg_order_value
  FROM sales_enriched
  GROUP BY order_month, country, category
)

SELECT *
FROM sales_summary
ORDER BY total_revenue DESC;