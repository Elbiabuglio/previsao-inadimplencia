from src.database import get_connection

if __name__ == "__main__":
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sys.databases;")
    for row in cursor.fetchall():
        print(row[0])

    conn.close()