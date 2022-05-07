import math
from os.path import join
from time import perf_counter_ns
import gc

dir = "tsp_dataset"

class MinHeap:
    """
    MinHeap Data Structure.
    It is not generalized, but focused only for Prim's algorithm. 
    """
    def __init__(self, arr) -> None:
        self.heapSize = len(arr)
        self.heap = []
        self.nodes = {}
        #Clone of the array. We don't want to change the original array
        for i in range(self.heapSize):
            self.heap.append(arr[i])
            #Map between a node and its index. 
            #Useful so the operation of checking if a node is in the heap has time O(1)
            self.nodes[arr[i]['node']] = i
        
        #First ordering, starting from the middle down to 1
        for i in reversed(range(self.heapSize // 2 )):
            self.minHeapify(i)


    def parent(self, i):
        return (i - 1) // 2
            
    def left(self, i):
        return (i * 2) + 1

    def right(self, i):
        return (i * 2) + 2

    def isLower(self, first, second):
        """
            Lower operation to handle cases with 'Inf' string value
        """
        if first == second:
            return False
        elif first == 'Inf':
            return False
        elif second == 'Inf':
            return True
        else:
            return first < second

    def isGreater(self, first, second):
        if first == second:
            return False
        elif first == 'Inf':
            return True
        elif second == 'Inf':
            return False
        else:
            return first > second
    

    def exchange(self, i, j):
        """
            Exchange element in position i with element in position j. 
            Update also the position of the nodes
        """
        self.nodes[self.heap[i]['node']] = j
        self.nodes[self.heap[j]['node']] = i

        tmp = self.heap[i]
        self.heap[i] = self.heap[j]
        self.heap[j] = tmp

    def minHeapify(self, i):
        """
            Maintain the property of the Min Heap, that is that
            the parent is always lower than the left and right child. 
            If it is not, then exchange the values and run again.
        """
        left = self.left(i)
        right = self.right(i)
        #Check if the left child exists and it is not lower then the parent 
        if left < self.heapSize and self.isLower(self.heap[left]['key'], self.heap[i]['key']):
            lowest = left
        else:
            lowest = i

        #Check if the right child exists and is not lower then the lowest between the parent and the left
        if right < self.heapSize and  self.isLower(self.heap[right]['key'], self.heap[lowest]['key']): 
            lowest = right

        #If the lowest is not the parent, fix the order and run again
        if lowest != i:
            self.exchange(i, lowest)
            self.minHeapify(lowest)
    
    def isEmpty(self):
        """
            Check if the Heap is empty
        """
        return self.heapSize == 0

    def hasNode(self, node):
        """
            Check if Heap has a given node
        """
        return self.nodes[node] < self.heapSize

    def minimum(self):
        """
            Get the minimum
        """
        return self.heap[0]

    def extractMin(self):
        """
            Extract the minimum and reorder the Heap
        """
        if self.heapSize < 1:
            print("Heap underflow")
            exit(1)
            
        min = self.heap[0]
        self.exchange(0, self.heapSize - 1)
        self.heapSize -= 1
        self.minHeapify(0)
        
        return min

    def getIndexByNode(self, node):
        """
            Get the index of a given node in the Heap if there is, otherwise -1
        """
        return self.nodes[node] if self.hasNode(node) else -1


    def updateNode(self, node, key, parent):
        """
            Update the data of an element and reoder the MinHeap if necessary
        """
        index = self.getIndexByNode(node)

        if index != -1:
            if self.isGreater(key, self.heap[index]['key']):
                print("New key is greater than current key")
                exit(1)

            self.heap[index]['key'] = key
            self.heap[index]['parent'] = parent

            #Push the node up in the tree if the new value is lower than the parent
            while index > 0 and not self.isLower(self.heap[self.parent(index)]['key'], self.heap[index]['key']):
                self.exchange(self.parent(index), index)
                index = self.parent(index)

def prim(graph, root):
    """
        Prim's algorithm with Heap Implementation for Minimum Spanning Trees.
        Graph is an adjacence list and root is the root of the tree
    """
    supp = {}
    #Initialization
    for node in graph.keys():
        supp[node] = {
            'node': node,
            'key': 'Inf',
            'parent': None
        }

    #Update root key to 0. The root will be the first to be extracted
    supp[root]['key'] = 0

    #Create MinHeap based on key value
    q = MinHeap(list(supp.values()))
    
    while not q.isEmpty():
        #Extract the node with minimum weight.
        minimum = q.extractMin()
        #For each node adjacent to minimum node
        for tuple in graph[minimum['node']]:
            v, weight = tuple
            #If the node has not been extracted yet, and the weight of the edge is
            #lower then node's current key, update the key and its parent
            if q.hasNode(v) and q.isLower(weight, supp[v]['key']):
                q.updateNode(v, weight, minimum['node'])
    
    return supp


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
        - Then a Preorder visit is performed on the result, to create a cycle touching all nodes.
        - Finally the starting point is added to the tail of the list, to close the cycle

        At the end we obtain an Hamiltonian cycle.
    """

    # Get MST
    mst = prim(graph, start)
    # Build a graph from the result of the MST
    graph_mst = {}
    for n in mst.keys():
        if mst[n]['parent'] != None:
            if graph_mst.get(mst[n]['parent']) == None:
                graph_mst[mst[n]['parent']] = []

            if graph_mst.get(mst[n]['node']) == None:
                graph_mst[mst[n]['node']] = []

            graph_mst[mst[n]['parent']].append((mst[n]['node'], mst[n]['key']))

    # Perform Preorder visit on the graph just build
    result = make_cycle(graph_mst, start, [start])
    # Add the start to the end to close the cycle
    result.append(start)
    return result

def get_cycle_nearest_neighbor(graph : dict, start: str):
    """
        Nearest Neighbor algorithm for TSP problem with log(n)-approximation. 
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

        # Find the node not in the circuit with the minimum distance from the last added node
        for elem in adj_list:
            n, w = elem
            if n not in result and (min == None or w < min):
                min = w
                min_node = n

        # Add the node to the circuit, and remove it from the available nodes
        result.append(min_node)
        keys.remove(min_node)
        iterator = min_node

    # Add the start at the end to close the cycle
    result.append(start)
    return result


def get_cycle_closest_insertion(graph: dict, start: str):
    """
        Closest Insertion algorithm for TSP problem, with a 2-factor approximation.
        For this algorithm an adjacent matrix is required as input. 
    """
    keys = list(graph.keys())
    result = [start]
    keys.remove(start)
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
        # Find the node in the circuit after which the new node will be inserted. 
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
        Find the node that minimize the distance from the circuit c
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
        The node in position i is returned.
    """
    i = 0
    min_node = None
    min = None
    while i < len(c) - 2:
        j = i + 1
        weight = graph[c[i]][k] + graph[k][c[j]] - graph[c[i]][c[j]]
        if min == None or weight < min:
            min = weight
            min_node = c[i]
        i += 1

    if min_node == None:
        min_node = c[0]

    return min_node

def get_result_sum(result, graph):
    """
        Given a list that represents an Hamiltonian cycle, 
        compute the total sum of the distances of the list.
    """
    sum = 0
    for i in range(len(result) - 1):
        node = result[i]
        next = result[i + 1]
        for elem in graph[node]:
            if(next == elem[0]):
                sum += elem[1]
    return sum

def measure_run_time(f, graph, root, num_calls, num_instances):
    """
        Compute the result and how much time takes to the function f to 
        compute on a graph starting from a given node. 
        The computation is repeated num_calls * num_instances time, to get 
        an average result of the execution time
    """
    sum_times = 0.0
    res = None
    for i in range(num_instances):
        gc.disable() #Disable garbage collector
        start_time = perf_counter_ns() 
        for i in range(num_calls):
            res = f(graph, root)
        end_time = perf_counter_ns()
        gc.enable()
        sum_times += (end_time - start_time)/num_calls
    avg_time = int(round(sum_times/num_instances))
    # return average time in nanoseconds and the result
    return avg_time, res

if __name__ == "__main__":
    
    # Files to be parsed
    files = [
        "burma14.tsp",  
        "ulysses16.tsp",
        "ulysses22.tsp",
        "eil51.tsp",
        "berlin52.tsp",
        "kroD100.tsp",
        "kroA100.tsp",
        "ch150.tsp",  
        "gr202.tsp",
        "gr229.tsp",
        "pcb442.tsp",
        "d493.tsp",
        "dsj1000.tsp",
    ]

    # Optimal solution for each file
    optimal_sol = [
        3323,
        6859,
        7013,
        426,
        7542,
        21294,
        21282,
        6528,
        40160,
        134602,
        50778,
        35002,
        18659688
    ]

    i = -1

    start_time = perf_counter_ns()
    for filename in files:
        i += 1
        print("Started file:", filename, 10 * "-") 
        nodes = []
        # Graph representation using adjancent lists
        graph = {}
        # Graph representation using adjacent matrix
        graph_matrix = {}
        file_graph = open(join(dir, filename))
        edge_type = None
        dimension = 0
        info = "None"

        # Collect some information from the first lines of the file,
        # such as the number of nodes and the type of the weight of the edges.
        while info and info != "NODE_COORD_SECTION":
            info = file_graph.readline().strip()
            if info != "NODE_COORD_SECTION":
                key, value = info.split(":")

                if key.strip() == "EDGE_WEIGHT_TYPE":
                    edge_type = value.strip()

                if key.strip() == "DIMENSION":
                    dimension = int(value.strip())
        
        # Exit in case of unsupported types
        if edge_type != "GEO" and edge_type != "EUC_2D":
            exit("Edge type not supported: expected [GEO, EUC_2D], got " + edge_type)

        # Lambda function to filter all empty strings
        filter_lambda = lambda s : len(s) > 0

        # Parse the nodes
        while info:
            info = file_graph.readline().strip()
            if info and info != "EOF":
                # N name of the node
                # X latitude
                # Y longitude
                (n, x, y) = list(filter(filter_lambda, info.split(" ")))

                x = float(x)
                y = float(y)

                # Transform the coordinates to radiants if GEO type
                if edge_type == "GEO":
                    x = to_radiants(x)
                    y = to_radiants(y)
                
                nodes.append((n, x, y))
        
        # Build the graphs
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

        """for n in graph_matrix.keys():
            print(n, graph[n])"""
        
        # We start always from root 1. 
        # Another option would be to iterate n-times as the number of nodes of the graph and then take 
        # in consideration only the best solution. 
        root = '1'

        print("Running 2-approx...")
        time, result = measure_run_time(get_cycle_2_approx, graph, root, 100, 10)
        sum = get_result_sum(result, graph)
        print("Optimal:", optimal_sol[i])
        print("Sum:", sum)
        print("Time(ns):", time)
        print("Error:", (sum - optimal_sol[i]) / optimal_sol[i])

        print("Running nearest neighbor...")
        time, result = measure_run_time(get_cycle_nearest_neighbor, graph, root, 100, 10)
        sum = get_result_sum(result, graph)
        print("Optimal:", optimal_sol[i])
        print("Sum:", sum)
        print("Time(ns):", time)
        print("Error:", (sum - optimal_sol[i]) / optimal_sol[i])
        
        print("Running closest insertion...")
        time, result = measure_run_time(get_cycle_closest_insertion, graph_matrix, root, 100, 10)
        sum = get_result_sum(result, graph)
        print("Optimal:", optimal_sol[i])
        print("Sum:", sum)
        print("Time(ns):", time)
        print("Error:", (sum - optimal_sol[i]) / optimal_sol[i])
        
        print(100 * "-")

    end_time = perf_counter_ns() 
    print("Finished all in:", (end_time - start_time) / 1000000000, "seconds")


        