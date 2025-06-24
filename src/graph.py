from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, overload, override
from heapq import heapify, heappush

from util import Least, Lexicographic, ifnone, todo_heap

_default = object()

class IGraph[K, E, V, Node](ABC):
    adj: dict[K, Node]

    ### Abstract methods for interfacing with the node ###

    @abstractmethod
    def _node(self, value: V) -> Node:
        '''Build a node with no edges for the graph.'''

    @abstractmethod
    def _setvalue(self, node: Node, value: V):
        '''Set the value of a node.'''
    
    @abstractmethod
    def _valueof(self, node: Node) -> V:
        '''Return the value of a node.'''

    @abstractmethod
    def _edges(self, node: Node) -> Iterable[tuple[K, E]]:
        '''Return the edges of a node.'''
    
    @abstractmethod
    def _add_edge(self, src: Node, dst: K, edge: E):
        '''Add edge to a node.'''

    @abstractmethod
    def _pop_edge(self, src: Node, dst: K) -> Optional[E]:
        '''Remove an edge from a node. If the edge does not exist, return None.'''

    ### Concrete methods ###

    def __init__(self, keys: dict[K, V]|None = None):
        super().__init__()
        self.adj = {k: self._node(v) for k, v in (keys or {}).items()}
    
    def __repr__(self):
        return f"{type(self).__name__}({self.adj!r})"
    
    def __contains__(self, key: K) -> bool:
        return key in self.adj
    
    def __bool__(self) -> bool:
        return bool(self.adj)
    
    def __len__(self) -> int:
        return len(self.adj)
    
    def __setitem__(self, key: K, value: V):
        '''Set the value of a node.'''
        if key in self.adj:
            self._setvalue(self.adj[key], value)
        else:
            self.adj[key] = self._node(value)

    def __getitem__(self, key: K) -> V:
        '''Get the value of a node.'''
        return self.get(key)
    
    def __iter__(self):
        return iter(self.adj)
    
    @overload
    def get(self, key: K, /) -> V: ...
    @overload
    def get[D](self, key: K, /, default: D) -> V|D: ...

    def get[D](self, key: K, /, default: D = _default) -> V|D:
        '''Get the value of a node, or return default if not found.'''
        if key in self.adj:
            return self._valueof(self.adj[key])
        if default is _default:
            raise KeyError(key)
        return default

    def insert(self, key: K, value: V):
        if key not in self.adj:
            self.adj[key] = self._node(value)

    @overload
    def edges(self, node: K, /) -> Iterable[tuple[K, E]]: ...
    @overload
    def edges[D](self, node: K, /, default: D) -> Iterable[tuple[K, E]]|D: ...
    
    def edges[D](self, node: K, /, default: D = _default) -> Iterable[tuple[K, E]]|D:
        '''Iterate over the edges of a node.'''
        n = self.adj.get(node, _default)
        if n is _default:
            if default is _default:
                raise KeyError(node)
            return default
        return self._edges(n) # type: ignore

    def add_edge(self, src: K, dst: K, edge: E):
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        if dst not in self.adj:
            raise KeyError(f"Destination {dst} not found")
        if src == dst:
            raise ValueError("Cannot add self-loop edge")
        
        self._add_edge(self.adj[src], dst, edge)

    def pop_edge(self, src: K, dst: K):
        '''
        Remove an edge from src to dst.
        If the edge does not exist, this is a no-op.
        '''
        if src not in self.adj:
            raise KeyError(f"Source {src} not found")
        
        return self._pop_edge(self.adj[src], dst)
    
    def keys(self) -> Iterable[K]:
        return self.adj.keys()
    
    def values(self) -> Iterable[V]:
        return map(self._valueof, self.adj.values())

    def items(self) -> Iterable[tuple[K, Node]]:
        return self.adj.items()

    def has_edge(self, src: K, dst: K) -> bool:
        '''Check if there is an edge from src to dst.'''
        return any(dst == k for k, _ in self.edges(src, []))

    def copy(self):
        '''Deepy copy of the graph.'''
        g = type(self)()
        for src, node in self.adj.items():
            g.insert(src, self._valueof(node))
            for dst, edge in self._edges(node):
                g.insert(dst, self._valueof(self.adj[dst]))
                g.add_edge(src, dst, edge)
        return g

    def invert(self):
        '''Invert the graph, reversing all edges.'''
        g = type(self)()
        for src, node in self.adj.items():
            g.insert(src, self._valueof(node))
            for dst, edge in self._edges(node):
                g.insert(dst, self._valueof(self.adj[dst]))
                g.add_edge(dst, src, edge)
        return g

    def toposort(self, key: Optional[Callable[[V], Optional[Lexicographic]]]=None) -> Iterable[K]:
        '''
        Kahn's algorithm for topological sorting.

        :param key: Optional function to determine the lexicographical order of nodes.
        '''

        if key is None:
            key = lambda v: None

        indeg = SimpleGraph[K, int]()
        for src in self:
            if src not in indeg:
                indeg.insert(src, 0)
            
            for dst, _ in self.edges(src):
                if dst not in indeg:
                    indeg.insert(dst, 1)
                else:
                    indeg[dst] += 1
                indeg.add_edge(src, dst, None)
        
        sources = [
            (ifnone(key(self[src]), Least), src)
                for src, deg in indeg.items()
                    if deg.value == 0
        ]
        heapify(sources)
        for _, src in todo_heap(sources):
            yield src
            for dst, _ in indeg.edges(src):
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    heappush(sources, (key(self[dst]), dst))

class SimpleNode[K, V]:
    value: V
    edges: set[K]

    def __init__(self, value: V):
        self.value = value
        self.edges = set()
    
    def __repr__(self):
        return f"SimpleNode(value={self.value}, edges={self.edges})"

class SimpleGraph[K, V](IGraph[K, None, V, SimpleNode[K, V]]):
    '''Simple graph using an extrusive node type with no edges.'''

    @override
    def _node(self, value: V) -> SimpleNode[K, V]:
        return SimpleNode(value)
    
    @override
    def _setvalue(self, node: SimpleNode[K, V], value: V):
        node.value = value
    
    @override
    def _valueof(self, node: SimpleNode[K, V]) -> V:
        return node.value
    
    @override
    def _edges(self, node: SimpleNode[K, V]) -> Iterable[tuple[K, None]]:
        for edge in node.edges:
            yield edge, None
    
    @override
    def _add_edge(self, src: SimpleNode[K, V], dst: K, edge: None):
        src.edges.add(dst)
    
    @override
    def _pop_edge(self, src: SimpleNode[K, V], dst: K) -> Optional[None]:
        if dst in src.edges:
            src.edges.remove(dst)
        return None

class Node[K, E, V]:
    '''Basic graph node.'''
    value: V
    edges: dict[K, E]

    def __init__(self, value: V):
        self.value = value
        self.edges = {}
    
    def __repr__(self):
        return f"Node(value={self.value}, edges={self.edges})"

class Graph[K, E, V](IGraph[K, E, V, Node[K, E, V]]):
    '''Easy default graph using an extrusive node type.'''

    @override
    def _node(self, value: V) -> Node[K, E, V]:
        return Node(value)
    
    @override
    def _setvalue(self, node: Node[K, E, V], value: V):
        node.value = value
    
    @override
    def _valueof(self, node: Node[K, E, V]) -> V:
        return node.value
    
    @override
    def _edges(self, node: Node[K, E, V]) -> Iterable[tuple[K, E]]:
        return node.edges.items()
    
    @override
    def _add_edge(self, src: Node[K, E, V], dst: K, edge: E):
        if dst in src.edges:
            raise ValueError(f"Edge to {dst} already exists in {src.value}")
        src.edges[dst] = edge
    
    @override
    def _pop_edge(self, src: Node[K, E, V], dst: K) -> Optional[E]:
        return src.edges.pop(dst)