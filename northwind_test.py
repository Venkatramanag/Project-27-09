import sqlite3
con = sqlite3.connect("northwind_core.sqlite")  # or northwind_core.db
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print([r[0] for r in cur.fetchall()])
