{{ config(indexes=[{'columns': ['customer_id'], 'unique': True}]) }}

SELECT
    {{ generate_key(['customer_id']) }} AS customer_key, 
    customer_id,
    first_name,
    last_name,
    first_name || ' ' || last_name AS full_name,
    email,
    registration_date,
    country,
    CASE 
        WHEN country IN ('USA', 'CAN') THEN 'North America'
        WHEN country IN ('UK', 'FRA', 'GER') THEN 'Europe'
        ELSE 'Rest of World'
    END AS customer_region
FROM {{ ref('stg_customers') }}