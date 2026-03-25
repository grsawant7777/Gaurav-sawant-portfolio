{{ config(indexes=[{'columns': ['product_id'], 'unique': True}]) }}

SELECT
    {{ generate_key(['product_id']) }} AS product_key,
    product_id,
    product_name,
    unit_price,
    supplier_id,
    CASE 
        WHEN category IN ('Electronics', 'Sports') THEN 'High Value'
        WHEN category IN ('Apparel', 'Books') THEN 'Standard Value'
        ELSE 'Other'
    END AS product_value_segment,
    category AS main_category
FROM {{ ref('stg_products') }}