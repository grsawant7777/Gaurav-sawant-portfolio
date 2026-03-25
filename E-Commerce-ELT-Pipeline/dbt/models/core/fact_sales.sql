{{
  config(
    materialized='incremental',
    unique_key='order_id',
    indexes=[
      {'columns': ['customer_key']},
      {'columns': ['product_key']},
      {'columns': ['date_key']}
    ]
  )
}}

WITH new_orders AS (
    -- PERFORMANCE OPTIMIZATION: Filter source data BEFORE joins
    SELECT * FROM {{ ref('stg_orders') }}
    
    {% if is_incremental() %}
      -- Handle late-arriving data or status updates with a 3-day buffer
      WHERE order_date >= (SELECT MAX(order_date) - interval '3 days' FROM {{ this }})
    {% endif %}
),

final AS (
    SELECT
        -- Use your custom macro to generate the integer date key
        {{ to_date_key('o.order_date') }} AS date_key,
        o.order_date, 
        c.customer_key,
        p.product_key, 
        o.total_amount AS order_total_amount,
        1 AS quantity_sold, 
        o.order_id,
        o.status,
        o.payment_method
    FROM new_orders o
    -- Joins are only performed on the subset of data defined in new_orders
    JOIN {{ ref('dim_customers') }} c ON o.customer_id = c.customer_id
    JOIN {{ ref('dim_products') }} p ON o.product_id = p.product_id
)

SELECT * FROM final