SELECT
    {{ clean_str('product_id') }} AS product_id,
    product_name,
    category,
    unit_price,
    supplier_id
FROM {{ source('ecom_raw', 'products') }}