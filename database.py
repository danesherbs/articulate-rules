import sqlite3
from typing import Optional
import pandas as pd

DATABASE = "database.db"


def execute_sql(sql_query: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(sql_query)


def create_fine_tuning_jobs_table():
    query = """CREATE TABLE IF NOT EXISTS fine_tuning_jobs (
                                model TEXT NOT NULL,
                                fpath TEXT NOT NULL,
                                job_id TEXT PRIMARY KEY NOT NULL
                             );"""
    execute_sql(query)


def job_id_exists(job_id: str) -> bool:
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fine_tuning_jobs WHERE job_id=?", (job_id,))
    count = cursor.fetchone()[0]
    return count > 0


def add_fine_tuning_job(model: str, fpath: str, job_id: str) -> None:
    if job_id_exists(job_id):
        return None

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO fine_tuning_jobs (model, fpath, job_id) VALUES (?, ?, ?)",
        (model, fpath, job_id),
    )
    conn.commit()


def remove_fine_tuning_job(job_id: str) -> None:
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM fine_tuning_jobs WHERE job_id = ?",
        (job_id,),
    )
    conn.commit()


def get_fine_tuning_job_id(model: str, fpath: str) -> str:
    """Returns the job id of the fine-tuning job for dataset at `fpath`."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT job_id FROM fine_tuning_jobs WHERE model = ? AND fpath = ?",
        (
            model,
            fpath,
        ),
    )
    result = cursor.fetchone()

    if result is None:
        raise ValueError(f"Job ID not found for model {model} and fpath {fpath}.")

    if len(result) >= 2:
        raise ValueError(f"Expected 1 result, got {len(result)}.")

    return result[0]


def create_labeled_articulations_table():
    query = """CREATE TABLE IF NOT EXISTS labeled_articulations (
                                expected_articulation TEXT NOT NULL,
                                actual_articulation TEXT NOT NULL,
                                is_equivalent BOOLEAN NOT NULL
                             );"""
    execute_sql(query)


def labeled_articulation_exists(
    expected_articulation: str, actual_articulation: str, is_equivalent: bool
) -> bool:
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM labeled_articulations WHERE expected_articulation = ? AND"
        " actual_articulation = ? AND is_equivalent = ?",
        (expected_articulation, actual_articulation, is_equivalent),
    )
    count = cursor.fetchone()[0]
    return count > 0


def add_labeled_articulation(
    expected_articulation: str,
    actual_articulation: str,
    is_equivalent: bool,
    expected_articulation_source: Optional[str] = None,
    actual_articulation_source: Optional[str] = None,
) -> None:
    if labeled_articulation_exists(
        expected_articulation, actual_articulation, is_equivalent
    ):
        return None

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO labeled_articulations (expected_articulation, actual_articulation,"
        " is_equivalent, expected_articulation_source, actual_articulation_source)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            expected_articulation,
            actual_articulation,
            is_equivalent,
            expected_articulation_source,
            actual_articulation_source,
        ),
    )
    conn.commit()


def remove_labeled_articulation(
    expected_articulation: str, actual_articulation: str
) -> None:
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM labeled_articulations WHERE expected_articulation = ? AND"
        " actual_articulation = ?",
        (expected_articulation, actual_articulation),
    )
    conn.commit()


def table_exists(table_name: str) -> bool:
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    result = cursor.fetchone()
    return result is not None


def get_table_as_df(table_name: str) -> pd.DataFrame:
    if not table_exists(table_name):
        raise ValueError(f"Table '{table_name}' does not exist in the database.")

    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    return df


def create_db():
    create_fine_tuning_jobs_table()
    create_labeled_articulations_table()


if __name__ == "__main__":
    create_db()
