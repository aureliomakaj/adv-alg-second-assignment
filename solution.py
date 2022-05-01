import math
from os.path import join
from select import select
from time import perf_counter_ns
import gc
import matplotlib.pyplot as plt
import prim

dir = "tsp_dataset"

def to_radiants(x):
    """
        Convert a geographic coordinate to radiants.
        Formula picked from:
        http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html
    """
    PI = 3.141592
    deg = int(x)
    min = x - deg
    return PI * (deg + 5 * min / 3) / 180

def distance(x1, y1, x2, y2):
    """
        Get the distance between two points represented through radiants coordinates.
        Formula picked from:
        http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html
    """
    RRR = 6378.388
    q1 = math.cos(y1 - y2)
    q2 = math.cos(x1 - x2)
    q3 = math.cos(x1 + x2)
    return int(RRR * math.acos( 0.5 * ((1 + q1) * q2 - (1 - q1 ) * q3) ) + 1)

def euclidian_distance(x1, y1, x2, y2):
    """
        Compute Euclidian Distance between two points
    """
    return math.trunc(math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def make_cycle(graph, n, list):
    """
        Preorder visit of a graph. 
        A list is used to store the nodes visited
    """
    for child in graph[n]:
        list.append(child[0])
        list = make_cycle(graph, child[0], list)
    return list


def get_cycle_2_approx(graph, start):
    """
        2-factor approximation algorithm for TSP problem, by using MST.
        - First a MST is found by using Prim's algorithm
        - Then a Preorder visits is performed on the result, to obtain the order in which the nodes are visited.
        - Finally the starting point is added to the tail of the list, to close the cycle

        At the end we obtain an Hamiltonian cycle, touching all nodes.
    """

    # Get MST
    mst = prim.prim(graph, start)

    # Build a graph from the result of the MST (which is a map with the nodes and foreach one the parent is specified)
    graph_mst = {}
    for n in mst.keys():
        graph_mst[n] = []
        for n2 in mst.values():
            if n2['parent'] == n:
                graph_mst[n].append((n2['node'], n2['key']))
    # Perform Preorder visit on the graph just build
    result = make_cycle(graph_mst, start, [start])
    # Add the start to the end to close the cycle
    result.append(start)
    return result

def get_cycle_nearest_neighbor(graph : dict, start: str):
    """
        Nearest Neighbor algorithm for TSP problem with a log(n)-approximation 
        First a list with the start node is created: foreach next step, the node with
        mimimum distance from the last element added is found.
    """
    keys = list(graph.keys())
    keys.remove(start)
    result = [start]
    iterator = start
    # Keep iterating until all nodes are added 
    while len(keys) > 0:
        adj_list = graph[iterator]
        min_node = None
        min = None

        # Fin the node not in the circuit with the minimum distance from the last added node
        for elem in adj_list:
            n, w = elem
            if n not in result and (min == None or w < min):
                min = w
                min_node = n

        # Add the node to the circuit, and remove it from the available nodes
        result.append(min_node)
        iterator = min_node
        keys.remove(iterator)

    # Add the start at the end to close the cycle
    result.append(start)
    return result


def get_cycle_closest_insertion(graph: dict, start: str):
    """
        Closest Insertion algorithm for TSP problem, with a 2-factor approximation.
    """
    keys = list(graph.keys())
    keys.remove(start)
    result = [start]
    # First step: find the node with minimum distance from the given start
    second = None
    min = None
    for node in graph[start].keys():
        if min == None or graph[start][node] < min:
            min = graph[start][node]
            second = node
    # Add the second node and remove it from the remaining nodes
    result.append(second)
    keys.remove(second)
    # Second step: iterate until there are no more nodes to be added
    while len(keys) > 0:
        # Find the node that minimze the distance from the circuit c
        min_vertex = select_min_vertex(graph, result, keys)
        # Find the node in the circuit after which the new node will be inserted . 
        node_to_insert = get_node_to_insert(graph, result, min_vertex)
        # Add the new node and remove it from the list
        index = result.index(node_to_insert)
        result.insert(index + 1, min_vertex)
        keys.remove(min_vertex)

    # Add the start node to close the cycle
    result.append(start)
    return result

def select_min_vertex(graph, c, rest):
    """
        Find the node that minimze the distance from the circuit c
    """
    min = None
    min_vertex = None
    for elem in rest:
        distance = circuit_vertex_distance(graph, c, elem)
        if min == None or distance < min:
            min = distance
            min_vertex = elem
    return min_vertex

def circuit_vertex_distance(graph, c, k):
    """
        Compute the minimum distance from a circuit c and a vertex k
    """
    min = None
    for h in c:
        if min == None or graph[h][k] < min:
            min = graph[h][k]
    return min

def get_node_to_insert(graph, c, k):
    """
        Given a circuit c and a vertex k, fin the edge (i, j) in the circuit
        where the distance is minimized by adding the vertex k between i and j.
        The node i is returned.
    """
    i = 1
    min_node = c[0]
    min = graph[min_node][c[1]]
    while i < len(c) - 2:
        j = i + 1
        """print(c[i], k, graph[c[i]][k])
        print(k, c[j], graph[k][c[j]])
        print(c[i], c[j], graph[c[i]][c[j]])
        print("++++")"""
        weight = graph[c[i]][k] + graph[k][c[j]] - graph[c[i]][c[j]]
        if weight < min:
            min = weight
            min_node = c[i]
        i += 1
    return min_node

if __name__ == "__main__":
    files = [
        "burma14.tsp",
        #"ulysses22.tsp",
        #"eil51.tsp",
        #"kroD100.tsp",
        #"gr229.tsp",
        #"d493.tsp",
        #"dsj1000.tsp"
    ]

    for filename in files:
        nodes = []
        graph = {}
        graph_matrix = {}
        file_graph = open(join(dir, filename))
        edge_type = None
        dimension = 0
        info = "None"
        while info and info != "NODE_COORD_SECTION":
            info = file_graph.readline().strip()
            if info != "NODE_COORD_SECTION":
                key, value = info.split(":")

                if key.strip() == "EDGE_WEIGHT_TYPE":
                    edge_type = value.strip()

                if key.strip() == "DIMENSION":
                    dimension = int(value.strip())
        
        if edge_type != "GEO" and edge_type != "EUC_2D":
            exit("Edge type not supported: expected [GEO, EUC_2D], got " + edge_type)

        filter_lambda = lambda s : len(s) > 0
        while info:
            info = file_graph.readline().strip()
            if info and info != "EOF":
                # N name of the node
                # X latitude
                # Y longitude
                (n, x, y) = list(filter(filter_lambda, info.split(" ")))

                x = float(x)
                y = float(y)
                if edge_type == "GEO":
                    x = to_radiants(x)
                    y = to_radiants(y)
                nodes.append((n, x, y))

        for tuple in nodes:
            (n1, x1, y1) = tuple
            graph[n1] = []
            graph_matrix[n1] = {}
            for tuple2 in nodes:
                (n2, x2, y2) = tuple2
                if n1 != n2:
                    if edge_type == "GEO":
                        w = distance(x1, y1, x2, y2)
                    else:
                        w = euclidian_distance(x1, y1, x2, y2)

                    graph[n1].append((n2, w))
                    graph_matrix[n1][n2] = w

        min = None
        min_root = 0
        for i in range(1, dimension):
            root = str(i)

            #result = get_cycle_2_approx(graph, root)
            #result = get_cycle_nearest_neighbor(graph, root)
            result = get_cycle_closest_insertion(graph_matrix, root)
            sum = 0
            for i in range(len(result) - 1):
                node = result[i]
                next = result[i + 1]
                for elem in graph[node]:
                    if(next == elem[0]):
                        sum += elem[1]
            #print(sum)
            if min == None or sum < min:
                min = sum
                min_root = root
        print("Best TSP: starting from", min_root, "with total", min)

        