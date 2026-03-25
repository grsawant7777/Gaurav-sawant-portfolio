{{
  config(
    materialized='table', 
    schema='analytics_mart',
    alias='product_category'
  )
}}

SELECT
    p.main_category,
    p.product_value_segment,
    COUNT(o.order_id) AS total_orders,
    SUM(o.order_total_amount) AS total_sales,
    AVG(o.order_total_amount) AS avg_order_value
FROM
    {{ ref('fact_sales') }} o
JOIN
    {{ ref('dim_products') }} p
    ON o.product_key = p.product_key
GROUP BY 1, 2
ORDER BY 4 DESC