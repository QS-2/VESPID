# invert the CITES
MATCH (p:Publication)-[rel:CITES]->(q:Publication)
CALL apoc.refactor.invert(rel)
yield input, output
RETURN input, output
# rename to CITED_BY
MATCH (p:Publication)-[rel:CITES]->(q:Publication)
WITH collect(rel) AS rels
CALL apoc.refactor.rename.type("CITES", "CITED_BY", rels)
YIELD batches, total, timeTaken, committedOperations
RETURN batches, total, timeTaken, committedOperations;

# set publication date to valid datetime type
MATCH (p:Publication) SET p.date = datetime(p.PublicationDate) RETURN p
# and wrote rel too
MATCH ()-[w:WROTE]-() SET w.date = datetime(w.PublicationDate) RETURN w

# query for test data
MATCH (i:Institution)-[f:EMPLOYED|AFFILIATED_WITH]-(a:Author)-[w:WROTE]->(p:Publication)-[r:CITED_BY]->(q:Publication)<-[x:WROTE]-(b:Author) return p, r, q, x, a, b, i, f LIMIT 25

# path expansion for test data
MATCH (p:Publication) WITH p CALL apoc.path.expand(p, "CITED_BY|WROTE|EMPLOYED|AFFILIATED_WITH", null, 1, 10) yield path return path limit 50

# in days duration for edge weights TODO put this in dynamic calculatoin for shortest path
MATCH (p:Publication)-[C:CITED_BY]->(q:Publication) return duration.inDays(p.date, q.date).days limit 5

CALL gds.graph.create.cypher(
    'citations',
    'MATCH (n:Publication) RETURN id(n) as id, labels(n) as labels',
    'MATCH (n:Publication)-[c:CITED_BY]->(m:Publication) RETURN id(n) AS source, id(m) AS target, c.days_duration as weight'
)

MATCH (n:Publication)-[:CITED_BY]->(m:Publication) RETURN id(n) AS source, id(m) AS target, duration.inDays(n.date, m.date).days as weight




CALL gds.graph.create.cypher(
    'citations',
    'MATCH (n:Publication) RETURN id(n) as id, labels(n) as labels',
    'MATCH (n:Publication)-[c:CITED_BY]->(m:Publication) RETURN id(n) AS source, id(m) AS target, c.days_duration as weight'
)

CALL gds.alpha.allShortestPaths.stream('citations', {relationshipWeightProperty: 'weight'}) YIELD sourceNodeId, targetNodeId, distance WHERE gds.util.isFinite(distance) = true AND distance > 0.0 return sourceNodeId, targetNodeId, distance order by distance DESC

CALL gds.alpha.allShortestPaths.stream('citations', {relationshipWeightProperty: 'weight'}) YIELD sourceNodeId, targetNodeId, distance WHERE gds.util.isFinite(distance) = true AND distance > 0.0 return gds.util.asNode(sourceNodeId).title as title, gds.util.asNode(sourceNodeId).authorNames as authors, gds.util.asNode(sourceNodeId).id as wos_id, gds.util.asNode(targetNodeId).title as target_title, gds.util.asNode(targetNodeId).authorNames as target_authors, gds.util.asNode(targetNodeId).id as target_wos_id, distance order by distance DESC
