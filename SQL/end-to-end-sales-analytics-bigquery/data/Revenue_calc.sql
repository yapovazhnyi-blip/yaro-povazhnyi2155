-- Revenue calculation
SELECT
  order_id,
  SAFE_MULTIPLY(CAST(quantity AS FLOAT64), CAST(unit_price AS FLOAT64)) AS revenue
FROM `main-project-712026.test.orders`;