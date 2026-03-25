SELECT
    {{ clean_str('customer_id') }} AS customer_id,
    first_name,
    last_name,
    email,
    registration_date,
    country
FROM {{ source('ecom_raw', 'customers') }}