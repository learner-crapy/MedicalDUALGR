import networkx as nx
import torch

# 创建一个有向多重图
hetero_graph = nx.MultiDiGraph()

# 添加不同类型的节点
hetero_graph.add_node("A1", type="Author")
hetero_graph.add_node("A2", type="Author")
hetero_graph.add_node("P1", type="Paper")
hetero_graph.add_node("P2", type="Paper")
hetero_graph.add_node("P3", type="Paper")
hetero_graph.add_node("V1", type="Venue")

# 添加不同类型的边
hetero_graph.add_edge("A1", "P1", type="writes")
hetero_graph.add_edge("A2", "P2", type="writes")
hetero_graph.add_edge("A1", "P3", type="writes")
hetero_graph.add_edge("P1", "V1", type="published_in")
hetero_graph.add_edge("P2", "V1", type="published_in")
hetero_graph.add_edge("P1", "P2", type="cites")
hetero_graph.add_edge("P2", "P3", type="cites")


def create_multi_adj_matrix(graph, edge_types):
    """
    为不同类型的边构建多邻接矩阵

    参数:
    graph: NetworkX 多重图对象
    edge_types: 需要构建邻接矩阵的边类型列表

    返回:
    adj_matrices: 多邻接矩阵字典
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}  # 映射节点到索引
    adj_matrices = {}

    for edge_type in edge_types:
        adj = torch.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵
        for u, v, data in graph.edges(data=True):
            if data['type'] == edge_type:
                u_idx, v_idx = node_index[u], node_index[v]
                adj[u_idx, v_idx] = 1  # 方向性邻接矩阵
        adj_matrices[edge_type] = adj

    return adj_matrices, node_index


# 定义所有边的类型
edge_types = ["writes", "published_in", "cites"]

# 计算多邻接矩阵
adj_matrices, node_index = create_multi_adj_matrix(hetero_graph, edge_types)
print(adj_matrices["published_in"])
