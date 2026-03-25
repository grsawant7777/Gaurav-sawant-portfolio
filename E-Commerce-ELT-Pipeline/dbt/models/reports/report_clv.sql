-- report_customer_lifetime_value.sql
{{
  config(
    materialized='table', 
    schema='analytics_mart',
    alias='clv'
  )
}}

SELECT
    c.customer_region,
    COUNT(DISTINCT c.customer_key) AS total_customers,
    SUM(o.order_total_amount) / COUNT(DISTINCT c.customer_key) AS customer_lifetime_value,
    COUNT(o.order_id) * 1.0 / COUNT(DISTINCT c.customer_key) AS average_orders_per_customer
FROM
    {{ ref('fact_sales') }} o
JOIN
    {{ ref('dim_customers') }} c 
    ON o.customer_key = c.customer_key
GROUP BY 1
ORDER BY 3 DESC