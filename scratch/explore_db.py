import os
import sys
import sqlite3
import argparse

DB_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"

def print_markdown_table(headers, rows):
    if not rows:
        print("No results found.")
        return
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))
            
    header_str = " | ".join(f"{h:<{col_widths[idx]}}" for idx, h in enumerate(headers))
    sep_str = "-|-".join("-" * col_widths[idx] for idx in range(len(headers)))
    print(f"| {header_str} |")
    print(f"| {sep_str} |")
    for row in rows:
        row_str = " | ".join(f"{str(val):<{col_widths[idx]}}" for idx, val in enumerate(row))
        print(f"| {row_str} |")

def main():
    parser = argparse.ArgumentParser(description="SQLite Database Explorer Utility")
    parser.add_argument("--query", "-q", type=str, help="SQL query to execute and print as a markdown table")
    parser.add_argument("--schema", "-s", action="store_true", help="Print the schema and size of all tables")
    parser.add_argument("--preview", "-p", type=str, help="Preview the first 5 rows of a table (e.g. activations, routing, metadata)")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if args.schema:
        print("=== Database Tables & Schemas ===")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"\nTable: {table} ({count} rows)")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            headers = ["Col ID", "Name", "Type", "NotNull", "Default", "PK"]
            print_markdown_table(headers, columns)

    elif args.preview:
        table = args.preview
        # Validate table name to prevent SQL injection
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        if table not in tables:
            print(f"Error: Table '{table}' does not exist. Choose from: {', '.join(tables)}")
            sys.exit(1)
            
        print(f"=== Previewing first 5 rows of table: {table} ===")
        cursor.execute(f"PRAGMA table_info({table})")
        headers = [col[1] for col in cursor.fetchall()]
        
        # Check if columns are large JSON arrays and truncate them for readability
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        rows = cursor.fetchall()
        clean_rows = []
        for r in rows:
            clean_r = []
            for val in r:
                val_str = str(val)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                clean_r.append(val_str)
            clean_rows.append(clean_r)
            
        print_markdown_table(headers, clean_rows)

    elif args.query:
        print(f"=== Executing Query: {args.query} ===")
        try:
            cursor.execute(args.query)
            if cursor.description:
                headers = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                clean_rows = []
                for r in rows:
                    clean_r = []
                    for val in r:
                        val_str = str(val)
                        if len(val_str) > 50:
                            val_str = val_str[:47] + "..."
                        clean_r.append(val_str)
                    clean_rows.append(clean_r)
                print_markdown_table(headers, clean_rows)
            else:
                conn.commit()
                print("Query executed successfully (no rows returned).")
        except Exception as e:
            print(f"Error: {e}")

    else:
        parser.print_help()

    conn.close()

if __name__ == "__main__":
    main()
