// Count papers cited by or citing each paper, grouped by the citation's cluster label
// This is a measure of how interdisciplinary the work is
MATCH (p:Publication)-[r:CITES]-(:Publication)
WITH p, COUNT(r) AS NumTotalCitations
MATCH (p:Publication)-[r:CITES]-(p2:Publication)
WITH p, NumTotalCitations, p2.pageCount AS ClusterLabel, COUNT(p2) AS NumCitationsInCluster
RETURN p.title, ClusterLabel, 
toFloat(NumCitationsInCluster) / NumTotalCitations AS FractionalMembership
//Note that this will need to be vectorized by python probably, such that zeroes are filled in for every missing cluster per paper