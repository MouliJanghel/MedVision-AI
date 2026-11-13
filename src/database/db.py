
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone


# Database location
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "medvision.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def current_time():
    return datetime.now(timezone.utc).isoformat()


# Initialize database tables
def init_db():
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                scan_type TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                gradcam_path TEXT,
                mask_path TEXT,
                review_status TEXT DEFAULT 'pending',
                reviewed_by TEXT,
                review_notes TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        # Prevent modification or deletion of audit records
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_audit_update
            BEFORE UPDATE ON audit_logs
            BEGIN
                SELECT RAISE(ABORT, 'Audit logs cannot be modified');
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
            BEFORE DELETE ON audit_logs
            BEGIN
                SELECT RAISE(ABORT, 'Audit logs cannot be deleted');
            END
        """)


# Save a new scan
def create_scan(
    filename,
    scan_type,
    prediction=None,
    confidence=None,
    gradcam_path=None,
    mask_path=None
):
    scan_id = str(uuid.uuid4())

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO scans (
                id, filename, scan_type, prediction,
                confidence, gradcam_path, mask_path,
                review_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            scan_id,
            filename,
            scan_type,
            prediction,
            confidence,
            gradcam_path,
            mask_path,
            current_time()
        ))

        conn.execute("""
            INSERT INTO audit_logs (scan_id, action, details, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            scan_id,
            "SCAN_CREATED",
            f"Scan uploaded: {filename}",
            current_time()
        ))

    return scan_id


# Approve or reject a scan
def review_scan(scan_id, decision, reviewed_by, notes=None):

    if decision not in ("approved", "rejected"):
        raise ValueError("Decision must be approved or rejected")

    with get_connection() as conn:

        cursor = conn.execute("""
            UPDATE scans
            SET review_status = ?,
                reviewed_by = ?,
                review_notes = ?,
                reviewed_at = ?
            WHERE id = ?
        """, (
            decision,
            reviewed_by,
            notes,
            current_time(),
            scan_id
        ))

        if cursor.rowcount == 0:
            raise ValueError("Scan not found")

        conn.execute("""
            INSERT INTO audit_logs (scan_id, action, details, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            scan_id,
            f"SCAN_{decision.upper()}",
            f"Reviewed by {reviewed_by}. Notes: {notes or 'None'}",
            current_time()
        ))


# Retrieve all scans
def get_scans():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM scans ORDER BY created_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


# Retrieve audit history
def get_audit_logs(scan_id=None):
    with get_connection() as conn:

        if scan_id:
            rows = conn.execute("""
                SELECT * FROM audit_logs
                WHERE scan_id = ?
                ORDER BY id DESC
            """, (scan_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM audit_logs
                ORDER BY id DESC
            """).fetchall()

        return [dict(row) for row in rows]
