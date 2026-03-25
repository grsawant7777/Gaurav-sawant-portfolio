{{
  config(
    materialized='table', 
    schema='analytics_mart',
    alias='daily_regional_sales'
  )
}}

SELECT
    order_date,
    customer_region,
    COUNT(o.order_id) AS daily_total_orders,
    SUM(o.order_total_amount) AS daily_total_sales 
FROM
    {{ ref('fact_sales') }} o
JOIN
    {{ ref('dim_customers') }} c 
    ON o.customer_key = c.customer_key
GROUP BY 1, 2
ORDER BY 1 DESC, 4 DESC