import networkx as nx
from graph.node import Node
import torch

def merge_into_new_nodes(level, old_nodes, graph):
    nodes = []
    for component in nx.connected_components(graph):
        node_info = (level, len(nodes))
        nodes.append(Node.create_segment_from_list([old_nodes[node] for node in component], node_info))
    return nodes


def update_graph(nodes, third_view_num, SEGMENT_CONNECT_RATIO):
    segment_frame_matrix = torch.stack([segment.frame for segment in nodes], dim=0)
    segment_frame_mask_matrix = torch.stack([segment.frame_mask for segment in nodes], dim=0)

    same_frame_matrix = torch.matmul(segment_frame_matrix, segment_frame_matrix.transpose(0,1))
    same_frame_mask_matrix = torch.matmul(segment_frame_mask_matrix, segment_frame_mask_matrix.transpose(0,1))

    disconnect_mask = torch.eye(len(nodes), dtype=bool).cuda()
    disconnect_mask = disconnect_mask | (same_frame_matrix < third_view_num)

    concensus_rate = same_frame_mask_matrix / (same_frame_matrix + 1e-7)
    A = concensus_rate >= SEGMENT_CONNECT_RATIO
    A = A & ~disconnect_mask
    A = A.cpu().numpy()

    G = nx.from_numpy_array(A)
    return G


def iterative_clustering(nodes, observer_num_thresholds, SEGMENT_CONNECT_RATIO, debug):
    if debug:
        print('====> Start iterative clustering')
    for i, observer_num_threshold in enumerate(observer_num_thresholds):
        if debug:
            print('observer_num', observer_num_threshold, 'number of nodes', len(nodes))
        graph = update_graph(nodes, observer_num_threshold, SEGMENT_CONNECT_RATIO)
        nodes = merge_into_new_nodes(i+1, nodes, graph)
    return nodes