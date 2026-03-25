"""
E-Commerce ELT Pipeline: Professional 7-Stage Medallion Architecture
Author: Gaurav Sawant
Stack: Airflow, dbt, PostgreSQL, Docker
"""

from datetime import timedelta
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os
import pandas as pd
from io import StringIO

# --- Configuration ---
POSTGRES_CONN_ID = "postgres_default"
DBT_PROJECT_DIR = "/opt/airflow/dbt"
RAW_DATA_PATH = "/opt/airflow/source_data"

INGESTION_MAP = {
    "customers": "raw_customers.csv",
    "products": "raw_products.csv",
    "orders": "raw_orders.csv",
}

# --- Functional Logic ---
def bulk_load_raw(table_name):
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    file_path = os.path.join(RAW_DATA_PATH, INGESTION_MAP[table_name])
    df = pd.read_csv(file_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, sep='\t')
    buffer.seek(0)
    
    conn = hook.get_conn()
    with conn.cursor() as cursor:
        cursor.execute(f"TRUNCATE TABLE raw_staging.{table_name} CASCADE;")
        cursor.copy_expert(f"COPY raw_staging.{table_name} FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t')", buffer)
    conn.commit()

def validate_bronze_data():
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    count = hook.get_first("SELECT COUNT(*) FROM raw_staging.orders")[0]
    if count == 0:
        raise ValueError("Bronze Quality Gate Failed: No records found in orders table.")

# --- DAG Definition ---
default_args = {
    'owner': 'Gaurav Sawant',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='Ecom_7_Stage_Medallion',
    default_args=default_args,
    start_date=days_ago(1),
    template_searchpath=['/opt/airflow/sql'],
    catchup=False,
    tags=['ELT', 'DBT','Medallion', 'DataEngineering']
) as dag:

    # STAGE 1: Infrastructure Setup (Bronze DDL)
    stage_1_init = PostgresOperator(
        task_id='stage_1_init_bronze_ddl',
        postgres_conn_id=POSTGRES_CONN_ID,
        sql='create_raw_tables.sql'
    )

    # STAGE 2: Parallel Ingestion (Bronze Load)
    stage_2_ingest = [
        PythonOperator(
            task_id=f'stage_2_load_{table}',
            python_callable=bulk_load_raw,
            op_kwargs={'table_name': table}
        ) for table in INGESTION_MAP
    ]

    # STAGE 3: Integrity Check (Bronze QA)
    stage_3_qa = PythonOperator(
        task_id='stage_3_bronze_integrity_gate',
        python_callable=validate_bronze_data
    )
    # STAGE 4: Normalization (Silver Core)
    stage_4_silver = BashOperator(
        task_id='stage_4_silver_transform_core',
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"dbt deps --profiles-dir . && "
            f"dbt run --select models/staging models/core --profiles-dir ."
        )
    )
 
    # STAGE 5: Aggregation (Gold Marts)
    stage_5_gold = BashOperator(
        task_id='stage_5_gold_transform_marts',
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"dbt run --select models/reports --profiles-dir ."
        )
    )

    # STAGE 6: Enterprise Validation (Final QA)
    stage_6_test = BashOperator(
        task_id='stage_6_dbt_test_and_audit',
        bash_command=f"cd {DBT_PROJECT_DIR} && dbt test"
    )

    # STAGE 7: Downstream Notification (Delivery)
    stage_7_ready = BashOperator(
        task_id='stage_7_pipeline_complete_signal',
        bash_command="echo 'E-commerce Pipeline Execution Successful. Data ready for BI consumption.'"
    )

    # --- Precise Dependency Graph ---
    stage_1_init >> stage_2_ingest >> stage_3_qa >> stage_4_silver >> stage_5_gold >> stage_6_test >> stage_7_ready