# List of algorithms

### Language model
1. Unigram model - bag of words, can be used for language identification
2. Ngram model - allow for phrase queries

### Boolean retrieval
1. Merge algorithm

### Postings list
1. Skip list
2. Positional index in postings list
3. Biword indexes - not as feasible as positional indexes

### Tolerant retrieval
#### Wildcard querying
1. Bigram indexes
2. Permuterm indexes
3. Search prefix trees
#### Spelling correction
1. Edit distance/ weighted edit distance
2. Jaccard distance
3. Ngram overlap
4. Hit based spelling correction
5. Context sensitive spelling correction

### Index construction
1. BSBI
2. SPIMI
3. Distributed indexing
4. Dynamic indexing
    - Logarithmic merge

### Index compression
1. Dictionary as a string
2. Blocked storage
3. Front coding
4. Postings compression

### Vector space model
1. Jaccard coefficient - bad, does not consider term frequencies
2. TF-IDF weighting
3. Cosine similarity - used for finding similarity between doc vs query

### Search heuristics
1. Faster cosine scoring - do addition of weights between query and doc instead for each term
2. Binary heap to retrieve top K
3. Index elimination
4. Champion list
5. Tiered indexes
6. Impact ordered posting
7. Cluster pruning
8. Adding additional info to contribute to document ranking

### IR Evaluation
#### Unranked eval, for unordered set of documents
1. F measures
#### Ranked eval
1. Precision at top K (widely used)
2. 11-point interpolated average precision
3. MAP
4. R-precision (describes only one point on the precision-recall curve, similar to 'Precision at k'
#### Creating test collections
1. Kappa measure - measure of agreement between judges

### Relevance feedback and query expansion
1. Rocchio expansion
2. Thesaurus generation

### XML retrieval
1. SimNoMerge - find context resemblance

### Probabilistic model
1. Okapi BM25
2. Language model search

### Web crawling and link analysis
1. PageRank (w or w/o teleportation)

