import psycopg2, json


# Manual debug script for Apache AGE Cypher parameters. Run it from inside
# the docker network (`docker compose exec loci-mcp python tests/...` is not
# possible because the tests dir is not mounted - copy it in or run from a
# container that reaches host `postgres`). Not a pytest test: pytest must
# not collect or import-execute it.

def probe(desc, sql, params=None):
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


def main():
    name_json = json.dumps("Tobias")
    print(f"JSON name: {repr(name_json)}")

    # Probe 1: Direct embed with JSON string in Cypher
    probe("Direct embed (fresh conn)",
        f"SELECT * FROM cypher('memory_graph', $$ MATCH (n {{name: {name_json}}}) RETURN n $$) AS (n agtype)")

    # Probe 2: Create a node
    probe("Create node",
        f"SELECT * FROM cypher('memory_graph', $$ CREATE (n:Person {{name: {name_json}}}) RETURN n $$) AS (n agtype)")

    # Probe 3: Now MATCH should find it (fresh conn)
    probe("Match after create (fresh conn)",
        f"SELECT * FROM cypher('memory_graph', $$ MATCH (n {{name: {name_json}}}) RETURN n $$) AS (n agtype)")

    # Probe 4: MATCH all
    probe("Match all",
        "SELECT * FROM cypher('memory_graph', $$ MATCH (n) RETURN n.name, labels(n) $$) AS (name agtype, labels agtype)")

    # Probe 5: Number value
    probe("Number param",
        "SELECT * FROM cypher('memory_graph', $$ RETURN 42 AS val $$) AS (val agtype)")

    # Probe 6: Relation
    probe("Create relation",
        f"SELECT * FROM cypher('memory_graph', $$ CREATE (a {{name: {json.dumps('Anna')}}}), (b {{name: {json.dumps('Bob')}}}) CREATE (a)-[:KNOWS]->(b) RETURN a, b $$) AS (a agtype, b agtype)")

    # Probe 7: Find connections
    probe("Find connections",
        f"SELECT * FROM cypher('memory_graph', $$ MATCH (a {{name: {name_json}}})-[r]-(b) RETURN b.name, type(r) $$) AS (neighbor agtype, rel agtype)")


if __name__ == "__main__":
    main()
