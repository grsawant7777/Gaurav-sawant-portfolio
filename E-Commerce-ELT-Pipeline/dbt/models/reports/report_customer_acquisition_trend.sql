-- report_customer_acquisition_trend.sql
{{
  config(
    materialized='table', 
    schema='analytics_mart',
    alias='customer_acquisition_trend'
  )
}}

SELECT
    DATE_TRUNC('month', registration_date) AS registration_month,
    customer_region,
    COUNT(customer_key) AS new_customers_count
FROM
    {{ ref('dim_customers') }}
GROUP BY 1, 2
ORDER BY 1 DESC, 3 DESC