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


if __name__ == "__main__":
    files = [
        "burma14.tsp",
        #"eil51.tsp"
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

        mst = prim.prim(graph, '1')
        
        graph_mst = {}
        for n in mst.keys():
            graph_mst[n] = []
            print(mst[n])
            for n2 in mst.values():
                if n2['parent'] == n:
                    graph_mst[n].append((n2['node'], n2['key']))
        """for k in graph.keys():
            print(k, graph[k])"""
        root = '1'
        result = make_cycle(graph_mst, root, [root])
        lastNode = result[len(result) - 1]
        result.append(graph[lastNode][0][0])
        sum = 0
        for i in range(len(result) - 1):
            node = result[i]
            next = result[i + 1]
            print(node, next)
            for elem in graph[node]:
                if(next == elem[0]):
                    print(node, next, elem[1])
                    sum += elem[1]
            print("----")
        print(sum)
        