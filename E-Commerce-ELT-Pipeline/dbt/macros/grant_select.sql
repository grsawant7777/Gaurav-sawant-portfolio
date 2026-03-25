{% macro grant_select(schema, role) %}

  {% set sql %}
    -- Ensure the user can see the schema
    GRANT USAGE ON SCHEMA {{ schema }} TO {{ role }};
    
    -- Ensure the user can read all current tables and views
    GRANT SELECT ON ALL TABLES IN SCHEMA {{ schema }} TO {{ role }};
    GRANT SELECT ON ALL SEQUENCES IN SCHEMA {{ schema }} TO {{ role }};
    
    -- Future-proofing: ensure new tables created later are also readable
    ALTER DEFAULT PRIVILEGES IN SCHEMA {{ schema }} 
    GRANT SELECT ON TABLES TO {{ role }};
  {% endset %}

  {% do log("Granting SELECT privileges on schema " ~ schema ~ " to role " ~ role, info=True) %}
  {% do run_query(sql) %}
  
{% endmacro %}