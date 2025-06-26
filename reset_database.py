"""
Simple script to reset the RAG evaluation database
"""

import sqlite3
import os

def reset_database(db_path="rag_evaluation.db"):
    """Reset the evaluation database"""
    try:
        if os.path.exists(db_path):
            # Connect and clear the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evaluations")
            conn.commit()
            conn.close()
            print(f"✅ Database '{db_path}' has been reset successfully!")
            print("All evaluation data has been cleared.")
        else:
            print(f"ℹ️ Database '{db_path}' doesn't exist yet.")
    except Exception as e:
        print(f"❌ Error resetting database: {e}")

if __name__ == "__main__":
    reset_database()
