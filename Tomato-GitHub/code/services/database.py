import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detection.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT, image_path TEXT, result_image_path TEXT,
        cls_en TEXT, cls_cn TEXT, conf REAL,
        severity TEXT, ratio REAL, advice TEXT)''')
    conn.commit()
    conn.close()


def save_record(time, image_path, result_image_path, cls_en, cls_cn,
                conf, severity, ratio, advice):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO records VALUES (NULL,?,?,?,?,?,?,?,?,?)",
              (time, image_path, result_image_path, cls_en, cls_cn,
               conf, severity, ratio, advice))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_records(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM records ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_record_by_id(record_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM records WHERE id = ?", (record_id,))
    row = c.fetchone()
    conn.close()
    return row


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT cls_cn, COUNT(*) FROM records GROUP BY cls_cn")
    cls_stats = c.fetchall()
    c.execute("SELECT severity, COUNT(*) FROM records GROUP BY severity")
    sev_stats = c.fetchall()
    conn.close()
    return cls_stats, sev_stats


def get_dashboard_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM records")
    total = c.fetchone()[0]

    c.execute("SELECT cls_cn, COUNT(*) FROM records GROUP BY cls_cn ORDER BY COUNT(*) DESC")
    cls_data = c.fetchall()

    c.execute("SELECT severity, COUNT(*) FROM records GROUP BY severity")
    sev_data = c.fetchall()

    trend = []
    today = datetime.now().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        c.execute("SELECT COUNT(*) FROM records WHERE time LIKE ?", (day_str + "%",))
        count = c.fetchone()[0]
        trend.append((day.strftime("%m-%d"), count))

    conn.close()

    return {
        "total": total,
        "cls_data": cls_data,
        "sev_data": sev_data,
        "trend": trend,
    }