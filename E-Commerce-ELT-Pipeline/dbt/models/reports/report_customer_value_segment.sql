{{
  config(
    materialized='table', 
    schema='analytics_mart',
    alias='customer_value_segment'
  )
}}

SELECT
    c.customer_region,
    COUNT(DISTINCT o.customer_key) AS total_customers,
    COUNT(o.order_id) AS total_orders,
    SUM(o.order_total_amount) AS total_sales,
    AVG(o.order_total_amount) AS avg_order_value
FROM
    {{ ref('fact_sales') }} o
JOIN
    {{ ref('dim_customers') }} c 
    ON o.customer_key = c.customer_key
GROUP BY 1
ORDER BY 4 DESC