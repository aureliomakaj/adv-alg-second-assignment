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


def build_graph(items):
    """
        Build a graph from a list of tuples of three elements: first node, second node, weight
    """
    graph = {}
    for item in items:
        n1, n2, w = item
        n1 = int(n1)
        n2 = int(n2)
        w = int(w)
        
        if graph.get(n1) == None:
            graph[n1] = []
        if graph.get(n2) == None:
            graph[n2] = []
        
        graph[n1].append((n2, w))
        graph[n2].append((n1, w))

    