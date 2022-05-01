import math
from os.path import join
from time import perf_counter_ns
import gc
from xml.etree.ElementPath import find
import matplotlib.pyplot as plt
import prim

dir = "tsp_dataset"

def to_radiants(x):
    PI = 3.141592
    deg = int(x)
    min = x - deg
    return PI * (deg + 5 * min / 3) / 180

def distance(x1, y1, x2, y2):
    RRR = 6378.388
    q1 = math.cos(y1 - y2)
    q2 = math.cos(x1 - x2)
    q3 = math.cos(x1 + x2)
    return int(RRR * math.acos( 0.5 * ((1 + q1) * q2 - (1 - q1 ) * q3) ) + 1)

def euclidian_distance(x1, y1, x2, y2):
    return math.trunc(math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def make_cycle(graph, n, list):
    for child in graph[n]:
        list.append(child[0])
        list = make_cycle(graph, child[0], list)
    return list


def get_cycle_2_approx(graph, root):
    mst = prim.prim(graph, root)
    graph_mst = {}
    for n in mst.keys():
        graph_mst[n] = []
        for n2 in mst.values():
            if n2['parent'] == n:
                graph_mst[n].append((n2['node'], n2['key']))
    result = make_cycle(graph_mst, root, [root])
    result.append(root)
    print(result)
    return result

def get_cycle_nearest_neighbor(graph : dict, root: str):
    keys = list(graph.keys())
    keys.remove(root)
    result = [root]
    iterator = root
    while keys and len(keys) > 0:
        adj_list = graph[iterator]
        min_node = None
        min = None
        for elem in adj_list:
            n, w = elem
            #print(n, w)
            if n not in result and (min == None or w < min):
                min = w
                min_node = n
        result.append(min_node)
        iterator = min_node
        keys.remove(iterator)

    result.append(root)
    return result


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
            for tuple2 in nodes:
                (n2, x2, y2) = tuple2
                if n1 != n2:
                    if edge_type == "GEO":
                        w = distance(x1, y1, x2, y2)
                    else:
                        w = euclidian_distance(x1, y1, x2, y2)

                    graph[n1].append((n2, w))

        min = None
        min_root = 0
        for i in range(1, dimension):
            root = str(i)

            #result = get_cycle_2_approx(graph, root)
            result = get_cycle_nearest_neighbor(graph, root)
            sum = 0
            for i in range(len(result) - 1):
                node = result[i]
                next = result[i + 1]
                for elem in graph[node]:
                    if(next == elem[0]):
                        sum += elem[1]
            print(sum)
            if min == None or sum < min:
                min = sum
                min_root = root
        print("Best TSP: starting from", min_root, "with total", min)

        