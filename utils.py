from pandas.core import indexing
from remote_server import *
import scipy.sparse as sp
import numpy as np
from pathlib import Path
import sys
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

    if args.graph == 'room':
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
        df.mid = df.mid
        df.subminid = df.subminid.apply(str)
        df.minid = df.minid.apply(str)
        df.majid = df.majid.apply(str)
        
        unique_mid = df.mid.unique()
        unique_subminid = df.subminid.unique()
        unique_minid = df.minid.unique()
        unique_majid = df.majid.unique()
        print(len(unique_mid),len(unique_subminid), len(unique_minid), len(unique_majid))
        
        node_mapping = {}
        index_mapping = {}
        for n in unique_mid:
            if n not in node_mapping:
                index_mapping[len(node_mapping)] = n 
                node_mapping[n] = len(node_mapping)
        
        subminid_mapping = {}
        subminid_index_mapping = {}
        for n in unique_subminid:
            if n not in subminid_mapping:
                subminid_index_mapping[len(subminid_mapping)] = n
                subminid_mapping[n] = len(subminid_mapping)
        
        minid_mapping = {}
        minid_index_mapping = {}
        offset = len(subminid_mapping)
        for n in unique_minid:
            if n not in minid_mapping:
                minid_index_mapping[len(minid_mapping)+offset] = n
                minid_mapping[n] = len(minid_mapping)+offset

        majid_mapping = {}
        majid_index_mapping = {}
        offset = len(subminid_mapping)+len(minid_mapping)
        for n in unique_majid:
            if n not in majid_mapping:
                majid_index_mapping[len(majid_mapping)+offset] = n
                majid_mapping[n] = len(majid_mapping)+offset

        emb = np.zeros(shape=(len(node_mapping), len(subminid_mapping)+len(minid_mapping)+len(majid_mapping)))
        print(emb.shape) # shape = (2493, 811)
        x = 1000
        y = 10
        for i in range(df.shape[0]):
            mid = df.iloc[i]['mid']
            subminid = df.iloc[i]['subminid']
            minid = df.iloc[i]['minid']
            majid = df.iloc[i]['majid']

            mid_idx = node_mapping[mid]
            subminid_idx = subminid_mapping[subminid]
            minid_idx = minid_mapping[minid]
            majid_idx = majid_mapping[majid]

            
            emb[mid_idx, subminid_idx] = x+np.random.randint(-y,y)
            emb[mid_idx, minid_idx] = 10*x
            emb[mid_idx, majid_idx] = 100*x
        np.set_printoptions(threshold=sys.maxsize)
        
        return emb, node_mapping, index_mapping




        # # Replace node names with indices 
        # df.mid = df.mid.map(node_mapping)
        # df.subminid = df.subminid.map(node_mapping)
        # df.minid = df.minid.map(node_mapping)
        # df.majid = df.majid.map(node_mapping)
        # df.root = df.root.map(node_mapping)
        # # Construct csr_matrix 
        # rows = np.concatenate([df.mid.values, df.subminid.values, df.minid.values, df.majid.values])
        # cols = np.concatenate([df.subminid.values, df.minid.values, df.majid.values, df.root.values])
        # data = np.ones(len(rows))
        # n_node = len(node_mapping)
        # adj_orig = sp.csr_matrix((data, (rows,cols)), shape = (n_node, n_node))
        # # Store the graph adjacency matrix
        # Path("data/").mkdir(parents=True, exist_ok=True)
        # np.savez("data/"+args.graph+"_graph.npz", adjacency = adj_orig, node_mapping = node_mapping, index_mapping = index_mapping)
        # print("Adjacency matrix is successfully generated!")
        # return adj_orig, node_mapping, index_mapping

    elif args.graph == 'doctor':
        df_doctor = get_sql_data(server, "select clip, sid from physicians")
        df_doctor.dropna(inplace = True)
        df_doctor = df_doctor[df_doctor['clip']!=""]
        unique_doctors = df_doctor['clip'].unique()
        unique_sids = df_doctor['sid'].unique()
        
        #nodes = nodes[nodes!='']
        node_mapping = {}
        index_mapping = {}
        for d in unique_doctors:
            if d not in node_mapping:
                index_mapping[len(node_mapping)] = d 
                node_mapping[d] = len(node_mapping)
        
        sid_mapping = {}
        sid_index_mapping = {}
        for s in unique_sids:
            if s not in sid_mapping:
                sid_index_mapping[len(sid_mapping)] = s 
                sid_mapping[s] = len(sid_mapping)
        # Update doctor ids with indices
        #df_doctor['clip'] = df_doctor['clip'].map(node_mapping)
        #grouped_df = df_doctor.groupby('sid')
        #grouped_lists = grouped_df['clip'].apply(list)
        #grouped_lists = grouped_lists.reset_index()
        #adj_orig = np.zeros((len(nodes), len(nodes)))
        #print(grouped_lists)
        x = 1000
        y = 10
        np.set_printoptions(threshold=sys.maxsize)
        emb = np.zeros(shape=(len(node_mapping), len(unique_sids)))
        for i in range(df_doctor.shape[0]):
            clip = df_doctor.iloc[i]['clip']
            sid = df_doctor.iloc[i]['sid']
            idx_clip = node_mapping[clip]
            idx_sid = sid_mapping[sid]
            emb[idx_clip, idx_sid] = x + np.random.randint(-y,y)
        return emb, node_mapping, index_mapping
    

def get_shortest_path_distances(args, adj):
    # Make sure the graph is undirected i.e. the matrix is symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    dist_mat = sp.csgraph.shortest_path(csgraph = adj, directed = False, return_predecessors = False, unweighted = args.unweighted)
    return dist_mat
