
DROP SCHEMA IF EXISTS raw_staging CASCADE; 
CREATE SCHEMA raw_staging;


CREATE TABLE raw_staging.customers (
    customer_id VARCHAR(50) PRIMARY KEY, 
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(100) UNIQUE, 
    registration_date DATE,
    country VARCHAR(100)
);


CREATE TABLE raw_staging.products (
    product_id VARCHAR(50) PRIMARY KEY, 
    product_name VARCHAR(255),
    category VARCHAR(100),
    unit_price DECIMAL(10, 2),
    supplier_id VARCHAR(50)
);


CREATE TABLE raw_staging.orders (
    order_id VARCHAR(50) PRIMARY KEY, 
    customer_id VARCHAR(50),
    product_id VARCHAR(50),
    order_date DATE,
    total_amount DECIMAL(10, 2),
    status VARCHAR(50),
    payment_method VARCHAR(50)
);