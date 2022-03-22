// Calculate the raw, non-normalized score for a given year and write it
// to the papers for that year only (and only the ones with full metadata 
// available, not the one-hop references with only IDs)
//      Assumes the graph projection desired exists
//      Examples given for year 2018
CALL gds.betweenness.stream(
      'citations_undirected_2018'
)
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
WHERE n.publicationDate.year = 2018
AND n.title IS NOT NULL
SET n.scoreInterDBWC_undirected_noSampling_raw = score;


// Given the un-normalized BWC values, normalize results
// using min-max scaling
MATCH (n:Publication)
WHERE n.publicationDate.year = 2018
AND n.scoreInterDBWC_undirected_noSampling_raw IS NOT NULL
WITH MIN(n.scoreInterDBWC_undirected_noSampling_raw) AS minimum, MAX(n.scoreInterDBWC_undirected_noSampling_raw) AS maximum

MATCH (n:Publication)
WHERE n.publicationDate.year = 2018
AND n.scoreInterDBWC_undirected_noSampling_raw IS NOT NULL
SET n.scoreInterDBWC_undirected_noSampling = (n.scoreInterDBWC_undirected_noSampling_raw - minimum) / (maximum - minimum);



// Generate undirected graph projection Louvain communities
CALL gds.louvain.stream(
'citations_undirected_2018'
)
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS n, communityId
WHERE n.publicationDate.year = 2018
SET n.louvainCommunityID = communityId;