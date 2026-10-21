import psycopg2, json

def test(desc, sql, params=None):
    conn = psycopg2.connect("postgresql://memory:geheim@postgres:5432/memory")
    cur = conn.cursor()
    cur.execute("LOAD 'age'")
    cur.execute("SET search_path TO ag_catalog, public")
    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        print(f"{desc}: {cur.fetchall()}")
    except Exception as e:
        print(f"{desc}: {e}")
    conn.close()

name_json = json.dumps("Tobias")
print(f"JSON name: {repr(name_json)}")

# Test 1: Direct embed with JSON string in Cypher
test("Direct embed (fresh conn)",
    f"SELECT * FROM cypher('memory_graph', $$ MATCH (n {{name: {name_json}}}) RETURN n $$) AS (n agtype)")

# Test 2: Create a node
test("Create node",
    f"SELECT * FROM cypher('memory_graph', $$ CREATE (n:Person {{name: {name_json}}}) RETURN n $$) AS (n agtype)")

# Test 3: Now MATCH should find it (fresh conn)
test("Match after create (fresh conn)",
    f"SELECT * FROM cypher('memory_graph', $$ MATCH (n {{name: {name_json}}}) RETURN n $$) AS (n agtype)")

# Test 4: MATCH all
test("Match all",
    "SELECT * FROM cypher('memory_graph', $$ MATCH (n) RETURN n.name, labels(n) $$) AS (name agtype, labels agtype)")

# Test 5: Number value
test("Number param",
    "SELECT * FROM cypher('memory_graph', $$ RETURN 42 AS val $$) AS (val agtype)")

# Test 6: Relation
test("Create relation",
    f"SELECT * FROM cypher('memory_graph', $$ CREATE (a {{name: {json.dumps('Anna')}}}), (b {{name: {json.dumps('Bob')}}}) CREATE (a)-[:KNOWS]->(b) RETURN a, b $$) AS (a agtype, b agtype)")

# Test 7: Find connections
test("Find connections",
    f"SELECT * FROM cypher('memory_graph', $$ MATCH (a {{name: {name_json}}})-[r]-(b) RETURN b.name, type(r) $$) AS (neighbor agtype, rel agtype)")
