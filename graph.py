from typing import Iterable, Optional, overload
from util import todo_list

class Graph[K, E, V]:
    class Node:
        value: V
        edges: dict[K, E]

        def __init__(self, value: V, edges: dict[K, E]):
            self.value = value
            self.edges = edges
        
        def __repr__(self):
            return f"Node(value={self.value}, edges={self.edges})"
    
    adj: dict[K, Node]

    def __init__(self, keys: dict[K, V]|None = None):
        super().__init__()
        self.adj = {k: Graph.Node(v, {}) for k, v in (keys or {}).items()}
    
    def __contains__(self, key: K) -> bool:
        return key in self.adj

    def insert(self, key: K, value: V):
        if key not in self.adj:
            self.adj[key] = Graph.Node(value, {})
    
    def __iter__(self):
        return iter(self.adj)

    def add_edge(self, src: K, dst: K, edge: E):
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        if dst not in self.adj:
            raise KeyError(f"Destination {dst} not found")
        if src == dst:
            raise ValueError("Cannot add self-loop edge")
        edges = self.adj[src].edges
        if dst in edges:
            raise ValueError(f"Edge from {src} to {dst} already exists")
        edges[dst] = edge
    
    def __getitem__(self, key: K) -> V:
        return self.adj[key].value
    
    def __setitem__(self, key: K, value: V):
        if key not in self.adj:
            raise KeyError(f"Key {key} not found")
        self.adj[key].value = value

    def pop_edge(self, src: K, dst: K):
        '''
        Remove an edge from src to dst.
        If the edge does not exist, this is a no-op.
        '''
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        
        return self.edges(src).pop(dst)
    
    @overload
    def edges(self, k: K) -> dict[K, E]: ...
    @overload
    def edges(self) -> Iterable[dict[K, E]]: ...

    def edges(self, k: Optional[K] = None) -> dict[K, E]|Iterable[dict[K, E]]:
        if k is not None:
            return self.adj[k].edges
        return (node.edges for node in self.adj.values())
    
    def has_edge(self, src: K, dst: K) -> bool:
        '''
        Check if there is an edge from src to dst.
        '''
        return dst in self.adj[src].edges

    def copy(self):
        '''
        Deepy copy of the graph.
        '''
        g = Graph[K, E, V]()
        for src, node in self.adj.items():
            g.insert(src, node.value)
            for dst, edge in node.edges.items():
                g.insert(dst, self.adj[dst].value)
                g.add_edge(src, dst, edge)
        return g

    def invert(self):
        '''
        Invert the graph, reversing all edges.
        '''
        g = Graph[K, E, V]()
        for src, node in self.adj.items():
            g.insert(src, node.value)
            for dst, edge in node.edges.items():
                g.insert(dst, self.adj[dst].value)
                g.add_edge(dst, src, edge)
        return g

    def toposort(self) -> Iterable[K]:
        '''
        Kahn's algorithm for topological sorting.
        '''

        indeg = Graph[K, None, int]()
        for src in self:
            if src not in indeg:
                indeg.insert(src, 0)
            
            for dst in self.edges(src):
                if dst not in indeg:
                    indeg.insert(dst, 1)
                else:
                    indeg[dst] += 1
                indeg.add_edge(src, dst, None)
        
        sources = [src for src in indeg if indeg[src] == 0]
        for src in todo_list(sources):
            yield src
            for dst in indeg.edges(src):
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    sources.append(dst)