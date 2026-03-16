# 🛍️ E-Commerce ELT Data Warehouse Pipeline

## Project: `ecommerce-elt-dwh`

### Project Status: Complete | Production Ready

| Skill | Technology | Badge |
| --- | --- | --- |
| **Orchestration** | Apache Airflow 2.7.2 |
| **Data Warehouse** | PostgreSQL 13-Alpine |
| **Transformation** | dbt (Data Build Tool) 1.7.5 |
| **Containerization** | Docker Compose 3.8 |
| **Data Modeling** | Medallion (7-Stage) | 
| **BI Layer** | Metabase |

---

## 🌟 1. Project Overview and Value Proposition

This project is a fully containerized **Extract, Load, Transform (ELT) pipeline** implementing a professional **7-Stage Medallion Architecture**. It automates the end-to-end flow of raw E-commerce data—from initial ingestion into a Bronze layer to the final delivery of validated, high-performance analytical marts in the Gold layer.

### Key Achievements (ATS Focus)

* **Modular Orchestration:** Engineered an idempotent 7-stage DAG in **Apache Airflow** using `PythonOperator` and `BashOperator` to ensure 100% automated recovery and scheduling.
* **Data as Code (DaC):** Centralized all transformation logic in **dbt**, utilizing Jinja macros for reusable code, surrogate key generation, and automated string cleansing.
* **Performance Optimization:** Developed an **Incremental Materialization** strategy in the Fact table, utilizing PostgreSQL indexing to support high-concurrency BI querying.
* **Data Governance:** Implemented automated quality gates, including a **Bronze Integrity Gate** and dbt-native referential integrity tests to prevent downstream data corruption.
* **Infrastructure as Code (IaC):** Orchestrated a multi-service container environment (Airflow, Postgres, Metabase) via **Docker Compose** for seamless deployment.

---

## 2. Solution Architecture and Data Flow

The architecture follows a standard ELT pattern, using **Airflow** to orchestrate the pipeline and **dbt** to define the transformation logic within **PostgreSQL**.

### 🔄 Detailed Project Workflow

The pipeline is strictly ordered into seven stages to maintain data consistency:

1. **Stage 1: Infrastructure Setup (DDL):** Initializes the warehouse environment and creates raw staging tables.
2. **Stage 2: Parallel Ingestion:** Bulk loads customers, products, and orders data into the Bronze layer using `PostgresHook`.
3. **Stage 3: Bronze Integrity Gate:** A Python-based validation check that verifies record counts and data health before transformation.
4. **Stage 4: Normalization (Silver):** dbt executes models to clean data (via `clean_str` macro) and builds the **Star Schema** core.
5. **Stage 5: Aggregation (Gold):** dbt generates business-ready marts (CLV, Category Sales, etc.) in the `analytics_mart` schema.
6. **Stage 6: Enterprise Validation:** Executes `dbt test` to audit for duplicates, nulls, and broken foreign key relationships.
7. **Stage 7: Consumption:** The Metabase dashboard queries the Gold tables to provide real-time KPI visibility.

---

## 3. Data Transformation Deep Dive (dbt)

The dbt project is structured to ensure clear separation of concerns and maintainable data lineage.

### A. Core Modeling: Star Schema

The `core` layer constructs the foundational analytical layer optimized for OLAP.

| Table Type | dbt Model | Key Logic & Optimizations |
| --- | --- | --- |
| **Fact Table** | `fact_sales.sql` | **Incremental** loading with a 3-day buffer; indexed on `date_key`, `customer_key`, and `product_key`. |
| **Dimension 1** | `dim_customers.sql` | Generates MD5 **Surrogate Keys** (`customer_key`) and derives geographic regions. |
| **Dimension 2** | `dim_date.sql` | Uses `to_date_key` macro to create integer-based date keys (e.g., `20260317`) for faster joins. |

### 🛠️ Data Transformation Deep Dive (dbt)

The transformation layer uses **dbt** to modularize the journey from raw data to high-performance analytical assets, focusing on scalability and strict quality.

#### A. Custom Macro Implementations

* **`clean_str`**: Enforces global standardization via `TRIM(LOWER())`.
* **`generate_key`**: Creates deterministic **MD5 Surrogate Keys** for stable schema relationships.
* **`to_date_key`**: Converts dates to **Integers** (e.g., `20260317`), replacing expensive date parsing with high-speed integer joins.

#### B. High-Performance Core Modeling (Silver Layer)

Transforms staging data into a foundational **Star Schema**.

* **Dimension Tables (`dim_`)**: Descriptive models (Customers, Products, Date) materialized with unique indexes for efficient filtering.
* **Incremental Fact Table (`fact_sales`)**: Processes only the last 3 days of data, reducing compute load by **~90%** while handling late-arriving records.

---

### 🥇 Analytical Marts (Gold Layer)

Materialized as physical tables in the `analytics_mart` schema for sub-second dashboard responses.

#### 1. Customer & Revenue Intelligence

* **`report_clv`**: Measures Customer Lifetime Value by region to identify top-tier geographic markets.
* **`report_acquisition_trend`**: Time-series analysis of monthly new registrations to track growth velocity.
* **`report_customer_value_segment`**: Regional performance tracking via Average Order Value (AOV) and sales volume.

#### 2. Sales & Operational Performance

* **`report_daily_regional_sales`**: A granular "pulse" view of daily revenue and volume per market.
* **`report_order_status`**: Tracks fulfillment health (`Pending`, `Shipped`, `Delivered`) to spot supply chain bottlenecks.

#### 3. Product & Payment Analytics

* **`report_product_category`**: Analyzes growth across categories (Electronics vs. Apparel) and value segments.
* **`report_payment_method`**: Correlates payment types (UPI, Credit Card) with spending habits and AOV.

#### 4. Data Governance & Security

* **Automated Testing**: Integrated `dbt tests` for null-checks and referential integrity (e.g., Sales ↔ Products).
* **Permission Automation**: Custom `grant_select` macro manages BI user access without manual DBA intervention.
## 4. Deployment and Setup

### Prerequisites

* Docker and Docker Compose installed.
* An `.env` file containing database secrets (provided in root).

### Quick Start

1. **Spin up the stack:**
```bash
docker-compose up --build -d

```


2. **Access Airflow:** Navigate to `localhost:8080`. (Default login: `airflow`/`airflow`).
3. **Trigger Pipeline:** Unpause the `Ecom_7_Stage_Medallion` DAG and manually trigger a run.
4. **Explore Data:** Open Metabase at `localhost:3000` to visualize the `analytics_mart` schema.

---

## 5. Repository Structure

```text
ecommerce-elt-dwh/
├── dags/                          <-- Airflow Orchestration
│   ├── ecom_elt_pipeline.py       <-- 7-Stage Medallion DAG
├── sql/                           <-- Schema & Table DDLs
├── dbt/                           <-- Data as Code (dbt Project)
│   ├── macros/                    <-- Custom SQL Logic (utils.sql, grant_select.sql)
│   ├── models/
│   │   ├── staging/               <-- Silver: Data Cleaning (Views)
│   │   ├── core/                  <-- Silver: Star Schema (Tables/Views)
│   │   └── reports/               <-- Gold: Business Marts (Tables)
│   └── dbt_project.yml
├── source_data/                   <-- Raw CSV Files (Ingestion Source)
├── docker-compose.yml             <-- Multi-service Orchestration
└── Dockerfile.airflow             <-- Custom Image (Airflow + dbt-postgres)

```

Would you like me to help you generate a `README.md` for your other projects, such as the **Healthcare 360 Lakehouse**?
