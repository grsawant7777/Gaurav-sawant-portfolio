-- report_payment_method_performance.sql
{{
  config(
    materialized='table', 
    schema='analytics_mart',
    alias='payment_method'
  )
}}

SELECT
    payment_method,
    COUNT(order_id) AS total_orders_count,
    SUM(order_total_amount) AS total_sales_volume,
    AVG(order_total_amount) AS avg_order_value
FROM
    {{ ref('fact_sales') }}
GROUP BY 1
ORDER BY 3 DESC