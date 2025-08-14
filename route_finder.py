import csv
import heapq
from typing import Dict, Tuple, List, Optional

class Graph:
    def __init__(self):
        # adjacency: node -> list of (neighbor, base_distance_km, traffic_percent)
        self.adj: Dict[str, List[Tuple[str, float, float]]] = {}

    def add_node(self, node: str) -> None:
        if node not in self.adj:
            self.adj[node] = []

    def add_edge(self, src: str, dst: str, distance_km: float, traffic_percent: float = 0.0, bidirectional: bool = True) -> None:
        """Add an edge. traffic_percent should be given as percentage (e.g., 20 for 20%)."""
        self.add_node(src)
        self.add_node(dst)
        self.adj[src].append((dst, float(distance_km), float(traffic_percent)))
        if bidirectional:
            self.adj[dst].append((src, float(distance_km), float(traffic_percent)))

    def neighbors(self, node: str):
        return self.adj.get(node, [])

    def nodes(self):
        return list(self.adj.keys())


def effective_distance_km(base_km: float, traffic_percent: float) -> float:
    """Return effective distance after traffic adjustment."""
    return base_km * (1.0 + traffic_percent / 100.0)


def dijkstra(graph: Graph, start: str, target: str) -> Optional[Tuple[float, List[str], float]]:
    """
    Dijkstra's algorithm using a min-heap (priority queue).
    Returns: (total_effective_distance, path_nodes_list, total_base_distance)
    If target unreachable, returns None.
    Complexity: O((V+E) log V)
    """
    if start not in graph.adj or target not in graph.adj:
        return None

    # distances measured as effective distance (base * (1+traffic%))
    dist: Dict[str, float] = {node: float('inf') for node in graph.adj}
    base_dist: Dict[str, float] = {node: float('inf') for node in graph.adj}
    prev: Dict[str, Optional[str]] = {node: None for node in graph.adj}

    dist[start] = 0.0
    base_dist[start] = 0.0
    pq: List[Tuple[float, str]] = [(0.0, start)]  # (distance, node)

    while pq:
        cur_d, node = heapq.heappop(pq)
        if cur_d > dist[node]:
            continue
        if node == target:
            break

        for (nei, base_km, traffic_pct) in graph.neighbors(node):
            eff = effective_distance_km(base_km, traffic_pct)
            cand = dist[node] + eff
            if cand < dist[nei]:
                dist[nei] = cand
                base_dist[nei] = base_dist[node] + base_km
                prev[nei] = node
                heapq.heappush(pq, (cand, nei))

    if dist[target] == float('inf'):
        return None

    # reconstruct path
    path: List[str] = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return dist[target], path, base_dist[target]


def load_graph_from_csv(path: str, bidirectional: bool = True) -> Graph:
    """
    CSV format: src,dst,distance_km,traffic_percent
    Header allowed but not required.
    """
    g = Graph()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # try to detect header
        rows = [r for r in reader if r and not all(cell.strip() == '' for cell in r)]
        for r in rows:
            if len(r) < 4:
                # skip malformed lines
                continue
            src, dst, dist_km, traffic = r[0].strip(), r[1].strip(), r[2].strip(), r[3].strip()
            try:
                d = float(dist_km)
                t = float(traffic)
            except ValueError:
                # maybe header line or bad row
                continue
            g.add_edge(src, dst, d, t, bidirectional=bidirectional)
    return g
