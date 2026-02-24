-- Basic row counts
SELECT COUNT(*) AS orders_rows FROM `main-project-712026.test.orders`;
SELECT COUNT(*) AS products_customers_rows FROM `main-project-712026.test.products_customers`;

-- Null checks (quick QA)
SELECT
  SUM(CASE WHEN order_id IS NULL THEN 1 ELSE 0 END) AS null_order_id,
  SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS null_customer_id,
  SUM(CASE WHEN product_id IS NULL THEN 1 ELSE 0 END) AS null_product_id
FROM `main-project-712026.test.orders`;