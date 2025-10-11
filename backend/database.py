import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_results (
        session_id TEXT PRIMARY KEY,
        prop_name TEXT,
        motor_name TEXT,
        esc_name TEXT,
        voltage REAL,
        csv_path TEXT,
        graph_paths TEXT,
        table_data TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_test_result(session_id, prop_name, motor_name, esc_name, voltage, csv_path, graph_paths, table_data):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO test_results (session_id, prop_name, motor_name, esc_name, voltage, csv_path, graph_paths, table_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            prop_name,
            motor_name,
            esc_name,
            voltage,
            csv_path,
            json.dumps(graph_paths),
            json.dumps(table_data)
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Error saving test result:", e)
        return False

def get_all_test_results():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test_results")
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append({
            "session_id": row[0],
            "prop_name": row[1],
            "motor_name": row[2],
            "esc_name": row[3],
            "voltage": row[4],
        })
    return results

def search_test_results(prop_name=None, motor_name=None, esc_name=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = "SELECT * FROM test_results WHERE 1=1"
    params = []
    if prop_name:
        query += " AND prop_name LIKE ?"
        params.append(f"%{prop_name}%")
    if motor_name:
        query += " AND motor_name LIKE ?"
        params.append(f"%{motor_name}%")
    if esc_name:
        query += " AND esc_name LIKE ?"
        params.append(f"%{esc_name}%")
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append({
            "session_id": row[0],
            "prop_name": row[1],
            "motor_name": row[2],
            "esc_name": row[3],
            "voltage": row[4],
        })
    return results

def get_test_result_by_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test_results WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "session_id": row[0],
        "prop_name": row[1],
        "motor_name": row[2],
        "esc_name": row[3],
        "voltage": row[4],
        "csv_path": row[5],
        "graph_paths": json.loads(row[6]),
        "table_data": json.loads(row[7]),
    }
