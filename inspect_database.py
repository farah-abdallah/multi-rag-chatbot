"""Simple database inspector to see the actual schema and data"""

import sqlite3
import os

def inspect_database():
    """Inspect the database structure and content"""
    
    # Find database files
    db_files = [f for f in os.listdir(".") if f.endswith(".db")]
    print(f"📁 Found database files: {db_files}")
    
    for db_file in db_files:
        print(f"\n🔍 Inspecting database: {db_file}")
        print("=" * 50)
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📋 Tables: {[table[0] for table in tables]}")
            
            # For each table, show schema and sample data
            for table_name, in tables:
                print(f"\n📊 Table: {table_name}")
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                print("   Columns:")
                for col in columns:
                    print(f"      {col[1]} ({col[2]}) - {('NOT NULL' if col[3] else 'NULL')}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"   Row count: {count}")
                
                # Show sample data
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    rows = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    print("   Sample data:")
                    for i, row in enumerate(rows):
                        print(f"      Row {i+1}:")
                        for j, value in enumerate(row):
                            print(f"         {column_names[j]}: {value}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error inspecting {db_file}: {e}")

if __name__ == "__main__":
    inspect_database()
