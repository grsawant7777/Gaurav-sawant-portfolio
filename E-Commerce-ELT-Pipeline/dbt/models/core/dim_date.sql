-- models/core/dim_date.sql
{{ config(materialized='table', schema='core') }}

WITH date_series AS (
    SELECT generate_series(
        '2021-01-01'::date, 
        '2025-12-31'::date, 
        '1 day'::interval
    )::date AS date_day
),

final AS (
    SELECT
        {{ to_date_key('date_day') }} AS date_key,
        date_day AS full_date,
        EXTRACT(YEAR FROM date_day) AS year,
        EXTRACT(QUARTER FROM date_day) AS quarter,
        EXTRACT(MONTH FROM date_day) AS month,
        TO_CHAR(date_day, 'Month') AS month_name,
        EXTRACT(DAY FROM date_day) AS day,
        -- Financial columns for better reporting
        CASE WHEN EXTRACT(MONTH FROM date_day) >= 4 THEN EXTRACT(YEAR FROM date_day)
             ELSE EXTRACT(YEAR FROM date_day) - 1 END AS fiscal_year
    FROM date_series
)

SELECT * FROM final