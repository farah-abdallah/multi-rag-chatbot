"""Debug script to test SQLite operations and check for sqlite_sequence table."""

import sqlite3
import os

def debug_sqlite_operations():
    """Test SQLite operations to identify the issue"""
    
    # Find the database file
    db_path = "evaluation_results.db"
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("Looking for database files...")
        for file in os.listdir("."):
            if file.endswith(".db"):
                print(f"Found database: {file}")
                db_path = file
                break
    
    print(f"📁 Using database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if evaluations table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations'")
        evaluations_table = cursor.fetchone()
        print(f"📋 Evaluations table exists: {evaluations_table is not None}")
        
        # Get record count
        if evaluations_table:
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            record_count = cursor.fetchone()[0]
            print(f"📊 Current record count: {record_count}")
        
        # Check if sqlite_sequence table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
        sequence_table = cursor.fetchone()
        print(f"🔢 sqlite_sequence table exists: {sequence_table is not None}")
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📚 All tables: {[table[0] for table in tables]}")
        
        # Test the problematic operation
        print("\n🧪 Testing delete operations...")
        
        # Test deleting from evaluations (safe)
        cursor.execute("DELETE FROM evaluations WHERE 1=0")  # Delete nothing
        print("✅ DELETE from evaluations works")
        
        # Test the sqlite_sequence operation safely
        print("🧪 Testing sqlite_sequence check...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
        if cursor.fetchone():
            print("✅ sqlite_sequence table found")
            # Test deletion (safe - where clause prevents actual deletion)
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='evaluations' AND 1=0")
            print("✅ DELETE from sqlite_sequence works")
        else:
            print("ℹ️ sqlite_sequence table not found (this is normal)")
        
        conn.close()
        print("✅ All operations completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")

if __name__ == "__main__":
    debug_sqlite_operations()
