
import os
import sys
from sqlalchemy import text

# Add the current directory to sys.path so we can import app
sys.path.append(os.getcwd())

# Force localhost for migration script running in shell
os.environ["DATABASE_URL"] = "postgresql://rechtmaschine:password@localhost:5432/rechtmaschine_db"

from app.database import SessionLocal

def migrate():
    print("Migrating database...")
    db = SessionLocal()
    try:
        # Check if column exists
        result = db.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='documents' AND column_name='gemini_file_uri'"))
        if result.fetchone():
            print("Column gemini_file_uri already exists.")
            return

        print("Adding gemini_file_uri column...")
        db.execute(text("ALTER TABLE documents ADD COLUMN gemini_file_uri VARCHAR(255)"))
        db.commit()
        print("Migration successful.")
    except Exception as e:
        print(f"Migration failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate()
