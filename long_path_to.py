import networkx as nx
import pandas as pd
import heapq
import math
import random
import matplotlib.animation
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

df = pd.read_csv("https://paste.sr.ht/blob/8ce2a2a7668a09cd852bc59f3987ef126ffe4bc3")

pub_dict = df.to_dict('index')
g = nx.Graph()
g.add_nodes_from(pub_dict)

pos_dict = {k: (v['lon'], v['lat']) for k,v in pub_dict.items()}
# print(len(g.nodes()))


def dist(idx1: int, idx2: int) -> float:
    (p1x,p1y) = pos_dict.get(idx1)
    (p2x,p2y) = pos_dict.get(idx2)
    dx = p2x - p1x
    dy = p2y - p1y
    return math.sqrt(dx * dx + dy * dy)


WALK_DIST = 0.02
g.clear_edges()
for k, v in pub_dict.items():
    edges = set()
    min_edge = None
    for k2, v2 in pub_dict.items():
        if k != k2 and dist(k, k2) < WALK_DIST:
            edges.add((k, k2, dist(k, k2)))
        if k != k2 and (min_edge is None or min_edge[2] > dist(k, k2)):
            min_edge = (k, k2, dist(k, k2))
    if len(edges) == 0:
        edges.add(min_edge)
    g.add_weighted_edges_from(edges)
# print(len(g.edges()))


def init_animation(g, pos):
    fig, ax = plt.subplots()
    return {'graph': g, 'pos': pos, 'fig': fig, 'ax': ax, 'frames': []}


def save_frame(an, title, visited, seen):
    an.get('frames').append({'title': title, 'visited': visited.copy(), 'seen': seen.copy()})


def draw_animation(an):
    G = an.get('graph')
    ax = an.get('ax')
    fig = an.get('fig')
    pos = an.get('pos')
    frames = an.get('frames')

    def update(num):
        ax.clear()
        frame = frames[num]
        title = frame.get('title')
        visited = frame.get('visited')
        seen = frame.get('seen')

        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="gray")
        null_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(G.nodes()) - set(visited) - set(seen), node_color="white",  ax=ax)
        null_nodes.set_edgecolor("black")


        vis_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(visited), node_color="black",  ax=ax)
        vis_nodes.set_edgecolor("black")

        seen_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(seen), node_color="gray", ax=ax)
        seen_nodes.set_edgecolor("white")
        # nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path,path)),  font_color="white", ax=ax)
        # edgelist = [path[k:k+2] for k in range(len(path) - 1)]
        # nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=idx_weights[:len(path)], ax=ax)

        ax.set_title("Frame %d:    "%(num+1) + " - " + title, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(frames), interval=1000, repeat=True)
    return HTML(ani.to_jshtml())


node_labels_dict = {k: k for k, _v in pub_dict.items()}
# nx.draw(g, pos=pos_dict, labels=node_labels_dict, node_size=20)

FROM = 41
TO = 85

# an = init_animation(g, nx.spring_layout(g))


def bfs(g, start):
    visited = {}
    visited[start] = True
    queue = [start]
    while queue:
        cur = queue.pop()
        save_frame(an, str(cur), visited, queue)
        for neighbour in g.neighbors(cur):
            if not visited.get(neighbour, False):
                visited[neighbour] = True
                queue.append(neighbour)


# bfs(g, FROM)
# draw_animation(an)

def dijkstra(g, start):
    distances = {node: float('inf') for node in g}
    distances[start] = 0

    priority_queue = [(0, start)]
    visited = set()
    seen = set([start])

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        save_frame(an, f"Visiting {current_node}", visited, seen)

        for neighbor in g.neighbors(current_node):
            distance = g.edges[current_node, neighbor]['weight']
            new_distance = current_distance + distance

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))
                seen.add(neighbor)

    return distances


# an = init_animation(g, pos_dict)
# dijkstra(g, FROM)
# print(draw_animation(an))

an = init_animation(g, pos_dict)


def astar(g, start, goal):
    def heuristic(node, goal):
        return dist(node, goal)

    open_set = [(0, start)]
    heapq.heapify(open_set)
    came_from = {}
    g_score = {node: float('inf') for node in g.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in g.nodes}
    f_score[start] = heuristic(start, goal)

    open_set_hash = {start}
    visited = []
    seen = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)
        save_frame(an, "Visiting {}".format(current), visited, seen)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        visited.append(current)

        for neighbor in g.neighbors(current):
            tentative_g_score = g_score[current] + g.edges[current, neighbor]['weight']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
                    seen.add(neighbor)

    return []


astar(g, FROM, TO)
print(draw_animation(an))