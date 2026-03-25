-- Macro to standardize string cleaning
{% macro clean_str(column_name) %}
    TRIM(LOWER({{ column_name }}))
{% endmacro %}

-- Macro to generate unique hashed keys
{% macro generate_key(field_list) %}
    md5(cast(concat(
        {% for field in field_list %}
            coalesce(cast({{ field }} as varchar), '_null_')
            {% if not loop.last %} , '-' , {% endif %}
        {% endfor %}
    ) as varchar))
{% endmacro %}

-- Macro to create integer date keys for performance
{% macro to_date_key(column_name) %}
    CAST(REPLACE(CAST({{ column_name }} AS TEXT), '-', '') AS INT)
{% endmacro %}