import sqlite3

conn = sqlite3.connect('test.db')
with open('sample_data.sql', 'r') as f:
    conn.executescript(f.read())
conn.close()