-- report_order_status_summary.sql
{{
  config(
    materialized='table', 
    schema='analytics_mart',
    alias='order_status_summary'
  )
}}

SELECT
    status,
    COUNT(order_id) AS order_count,
    SUM(order_total_amount) AS total_sales_volume
FROM
    {{ ref('fact_sales') }}
GROUP BY 1
ORDER BY 2 DESC