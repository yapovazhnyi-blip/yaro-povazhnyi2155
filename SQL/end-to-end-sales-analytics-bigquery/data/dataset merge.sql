SELECT
  o.order_id,
  DATE(o.order_date) AS order_date,
  o.customer_id,
  p.product_name,
  p.category,
  COALESCE(p.country, 'Unknown') AS country,
  o.quantity,
  o.unit_price,
  SAFE_MULTIPLY(CAST(o.quantity AS FLOAT64), CAST(o.unit_price AS FLOAT64)) AS revenue
FROM `main-project-712026.test.orders` o
LEFT JOIN `main-project-712026.test.products_customers` p
  ON o.product_id = p.product_id
 AND o.customer_id = p.customer_id;