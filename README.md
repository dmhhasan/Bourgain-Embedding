# Bourgain-Embedding for Graph
This repository contains the implementation of generating embedding for a undirected graph based on Bourgain's embedding technique. 

### Input
1. A graph adjacency matrix ```A = (NxN)```

### Output
1. Embeddings of the graph generally a matrix of shape, ```(NxM)``` where ```M =c.logN^2``` ( where ```c``` is a constant)


### Requirements
- Python 3.7.4
- numpy
- scipy
- mysql.connector
- pandas

### Run the demo
```python main.py --graph medication --c 2 --unweighted True``` <br /> <br />
If ```unweighted=True``` the embedding uses edge count instead of the edge weights.

### Directories
#### Data
- Graph adjacency matrix: ```/data/medication_graph.npz``` which contains three files "adjacency", "node_mapping", and "index_mapping"

#### Generated output
- Embedding of the graph: ```/output/medication_embedding.npz```; it also contains 3 files: "embedding", "node_mapping", and "index_mapping"

