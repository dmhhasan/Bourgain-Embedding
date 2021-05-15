from remote_server import *
import scipy.sparse as sp
import numpy as np
from pathlib import Path

def load_graph(args):
    file = Path("data/"+args.graph+"_graph.npz")
    if file.is_file():
        data = np.load(file, allow_pickle = True)
        adj, node_mapping, index_mapping = data['adjacency'][()], data['node_mapping'][()], data['index_mapping'][()]
        return adj, node_mapping, index_mapping
    else:
        return generate_graph(args)




def generate_graph(args):
    server = initialize_server(args)

    if args.graph == 'uihc':
        df = get_sql_data(server, "select srid, drid,cost from edges")
        df.dropna(inplace=True)
        # Minimum edge weight: 0.8 and maximum edge weigth = 5
        # Find unique room ids
        room_ids = pd.unique(df[['srid', 'drid']].values.ravel('K'))
        node_mapping = {}
        index_mapping = {}
        for r in room_ids:
            if r not in node_mapping:
                index_mapping[len(node_mapping)] = r
                node_mapping[r] = len(node_mapping)

        # Update room ids with indices
        df.srid = df.srid.map(node_mapping)
        df.drid = df.drid.map(node_mapping)

        # Create csr_matrix from the edges
        n_node = len(node_mapping)
        adj_orig = sp.csr_matrix((df.cost, (df.srid, df.drid)), shape=(n_node, n_node))

        # Store the graph adjacency matrix
        Path("data/").mkdir(parents=True, exist_ok=True)
        np.savez("data/"+args.graph+"_graph.npz", adjacency = adj_orig, node_mapping = node_mapping, index_mapping = index_mapping)
        print("Adjacency matrix is successfully generated!")
        
        return adj_orig, node_mapping, index_mapping


    elif args.graph == 'medication':
        df_med = get_sql_data(server, "select mid, subminid from medications")
        df_med.dropna(inplace = True)
        df_rx = get_sql_data(server, "select subminid, minid, majid from rxclass")
        df_rx.dropna(inplace=True)
        
        df = pd.merge(df_med, df_rx, on = "subminid")
        #  Avoid overlapping by prepending type
        df.mid = "mid_"+df.mid 
        df.subminid = "subminid_"+df.subminid.apply(str)
        df.minid = "minid_"+df.minid.apply(str)
        df.majid = "majid_"+df.majid.apply(str)
        df['root'] = "root_0"
        
        nodes = pd.unique(df[['mid', 'subminid', 'minid', 'majid', 'root']].values.ravel('K'))

        node_mapping = {}
        index_mapping = {}
        for n in nodes:
            if n not in node_mapping:
                index_mapping[len(node_mapping)] = n 
                node_mapping[n] = len(node_mapping)
        
        # Replace node names with indices 
        df.mid = df.mid.map(node_mapping)
        df.subminid = df.subminid.map(node_mapping)
        df.minid = df.minid.map(node_mapping)
        df.majid = df.majid.map(node_mapping)
        df.root = df.root.map(node_mapping)
        # Construct csr_matrix 
        rows = np.concatenate([df.mid.values, df.subminid.values, df.minid.values, df.majid.values])
        cols = np.concatenate([df.subminid.values, df.minid.values, df.majid.values, df.root.values])
        data = np.ones(len(rows))
        n_node = len(node_mapping)
        adj_orig = sp.csr_matrix((data, (rows,cols)), shape = (n_node, n_node))
        # Store the graph adjacency matrix
        Path("data/").mkdir(parents=True, exist_ok=True)
        np.savez("data/"+args.graph+"_graph.npz", adjacency = adj_orig, node_mapping = node_mapping, index_mapping = index_mapping)
        print("Adjacency matrix is successfully generated!")
        return adj_orig, node_mapping, index_mapping



def get_shortest_path_distances(args, adj):
    file = Path("dist_matrix/"+args.graph+"_dist_mat.npz")
    if file.is_file():
        return np.load(file)['dist_mat']
    else:
        # Make sure the graph is undirected i.e. the matrix is symmetric
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        dist_mat = sp.csgraph.shortest_path(csgraph = adj, directed = False, return_predecessors = False, unweighted = True)
        # Save distanc matrix
        Path("dist_matrix/").mkdir(parents=True, exist_ok=True)
        np.savez("dist_matrix/"+args.graph+"_dist_mat.npz", dist_mat = dist_mat)
        return dist_mat
