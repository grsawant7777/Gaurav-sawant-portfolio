SELECT
    order_id,
    {{ clean_str('customer_id') }} AS customer_id, 
    {{ clean_str('product_id') }} AS product_id,
    order_date,
    total_amount,
    status,
    payment_method
FROM {{ source('ecom_raw', 'orders') }}