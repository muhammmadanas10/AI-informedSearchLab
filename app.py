import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

st.set_page_config(
    page_title="Informed Search Visualizer", layout="wide"
)

st.title("Interactive Route Planning & Search Visualizer")
st.write(
    "Explore Greedy Best-First Search (GBFS) and A* Search on real-world route"
    " graphs."
)

# ---------------------------------------------------------
# Graph Data Setup
# ---------------------------------------------------------
graph_options = {
    "Warehouse Robot (Task 1)": {
        "locations": {
            "Receiving_Area": (0, 0),
            "Storage_A": (2, 1),
            "Storage_B": (1, 4),
            "Sorting_Area": (4, 2),
            "Inspection_Area": (5, 5),
            "Packing_Station": (7, 6),
        },
        "edges": [
            ("Receiving_Area", "Storage_B", 4.1),
            ("Receiving_Area", "Storage_A", 2.2),
            ("Storage_B", "Inspection_Area", 5.0),
            ("Storage_B", "Sorting_Area", 6.0),
            ("Storage_A", "Sorting_Area", 2.2),
            ("Sorting_Area", "Inspection_Area", 3.2),
            ("Sorting_Area", "Packing_Station", 5.0),
            ("Inspection_Area", "Packing_Station", 2.2),
        ],
        "default_start": "Receiving_Area",
        "default_goal": "Packing_Station",
    },
    "Airport Baggage Cart (Task 2)": {
        "locations": {
            "Baggage_Area": (0, 0),
            "Security": (2, 1),
            "Checkpoint": (1, 4),
            "Food_Court": (4, 2),
            "Terminal_Hall": (5, 5),
            "Departure_Gate": (8, 6),
        },
        "edges": [
            ("Baggage_Area", "Checkpoint", 4.1),
            ("Baggage_Area", "Security", 2.2),
            ("Checkpoint", "Terminal_Hall", 5.0),
            ("Security", "Food_Court", 2.2),
            ("Food_Court", "Terminal_Hall", 3.2),
            ("Food_Court", "Departure_Gate", 6.0),
            ("Terminal_Hall", "Departure_Gate", 3.2),
        ],
        "default_start": "Baggage_Area",
        "default_goal": "Departure_Gate",
    },
    "Hospital Emergency Robot (Task 3)": {
        "locations": {
            "Pharmacy": (0, 0),
            "Main_Corridor": (2, 1),
            "Patient_Wing": (1, 4),
            "Nursing_Station": (4, 2),
            "Laboratory": (5, 5),
            "Emergency_Ward": (8, 6),
        },
        "edges": [
            ("Pharmacy", "Patient_Wing", 4.1),
            ("Pharmacy", "Main_Corridor", 2.2),
            ("Patient_Wing", "Laboratory", 5.0),
            ("Main_Corridor", "Nursing_Station", 2.2),
            ("Nursing_Station", "Laboratory", 3.2),
            ("Nursing_Station", "Emergency_Ward", 6.0),
            ("Laboratory", "Emergency_Ward", 3.2),
        ],
        "default_start": "Pharmacy",
        "default_goal": "Emergency_Ward",
    },
}

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

selected_graph_name = st.sidebar.selectbox(
    "Select Scenario/Graph", list(graph_options.keys())
)
selected_graph = graph_options[selected_graph_name]

locations = selected_graph["locations"]
node_list = list(locations.keys())

start_node = st.sidebar.selectbox(
    "Select Initial Node (Start)",
    node_list,
    index=node_list.index(selected_graph["default_start"]),
)
goal_node = st.sidebar.selectbox(
    "Select Goal Node",
    node_list,
    index=node_list.index(selected_graph["default_goal"]),
)
algorithm = st.sidebar.selectbox(
    "Select Search Algorithm", ["Greedy Best-First Search (GBFS)", "A* Search"]
)


# Build NetworkX DiGraph
G = nx.DiGraph()
for node, pos in locations.items():
  G.add_node(node, pos=pos)
for u, v, weight in selected_graph["edges"]:
  G.add_edge(u, v, weight=weight)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def heuristic(node, target):
  x1, y1 = locations[node]
  x2, y2 = locations[target]
  return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def run_gbfs(graph, start, goal):
  pq = []
  count = 0
  heapq.heappush(pq, (heuristic(start, goal), count, start, [start], 0.0))
  visited = set()
  expansion_sequence = []

  while pq:
    h, _, current, path, cost = heapq.heappop(pq)
    if current in visited:
      continue
    visited.add(current)
    expansion_sequence.append(current)

    if current == goal:
      return expansion_sequence, path, cost

    for neighbor in graph.neighbors(current):
      if neighbor not in visited:
        edge_cost = graph[current][neighbor]["weight"]
        count += 1
        heapq.heappush(
            pq,
            (
                heuristic(neighbor, goal),
                count,
                neighbor,
                path + [neighbor],
                cost + edge_cost,
            ),
        )
  return expansion_sequence, [], 0.0


def run_a_star(graph, start, goal):
  pq = []
  count = 0
  g_costs = {node: float("inf") for node in graph.nodes()}
  g_costs[start] = 0.0
  f_costs = {node: float("inf") for node in graph.nodes()}
  f_costs[start] = heuristic(start, goal)

  heapq.heappush(pq, (f_costs[start], count, 0.0, start, [start]))
  visited = set()
  expansion_sequence = []

  while pq:
    f, _, g, current, path = heapq.heappop(pq)
    if current in visited:
      continue
    visited.add(current)
    expansion_sequence.append(current)

    if current == goal:
      return expansion_sequence, path, g

    for neighbor in graph.neighbors(current):
      weight = graph[current][neighbor]["weight"]
      tentative_g = g + weight
      if tentative_g < g_costs[neighbor]:
        g_costs[neighbor] = tentative_g
        f_costs[neighbor] = tentative_g + heuristic(neighbor, goal)
        count += 1
        heapq.heappush(
            pq, (f_costs[neighbor], count, tentative_g, neighbor, path + [neighbor])
        )
  return expansion_sequence, [], float("inf")


# ---------------------------------------------------------
# Run Selected Search Algorithm
# ---------------------------------------------------------
if start_node == goal_node:
  st.warning("Start node and Goal node are the same!")
else:
  if algorithm == "Greedy Best-First Search (GBFS)":
    expansion_seq, solution_path, total_cost = run_gbfs(
        G, start_node, goal_node
    )
  else:
    expansion_seq, solution_path, total_cost = run_a_star(
        G, start_node, goal_node
    )

  # Display Visualization
  fig, ax = plt.subplots(figsize=(10, 6))
  pos = locations
  path_edges = (
      list(zip(solution_path[:-1], solution_path[1:])) if solution_path else []
  )

  node_colors = []
  for node in G.nodes():
    if node == start_node:
      node_colors.append("lightcoral")
    elif node == goal_node:
      node_colors.append("gold")
    elif node in solution_path:
      node_colors.append("lightgreen")
    else:
      node_colors.append("lightblue")

  nx.draw_networkx_nodes(G, pos, node_size=2200, node_color=node_colors, ax=ax)
  nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
  nx.draw_networkx_edges(
      G,
      pos,
      edgelist=G.edges(),
      arrowstyle="->",
      arrowsize=20,
      edge_color="gray",
      ax=ax,
  )

  if path_edges:
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=path_edges,
        arrowstyle="->",
        arrowsize=25,
        edge_color="green",
        width=3,
        ax=ax,
    )

  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

  plt.title(
      f"{selected_graph_name} — {algorithm}", fontsize=14, fontweight="bold"
  )
  plt.xlabel("X Coordinates")
  plt.ylabel("Y Coordinates")
  plt.grid(True, linestyle="--", alpha=0.5)
  ax.set_axis_on()

  st.pyplot(fig)

  # Output Summary Details
  st.subheader("📊 Execution Results")
  col1, col2, col3 = st.columns(3)
  col1.metric("Selected Algorithm", algorithm.split()[0])
  col2.metric(
      "Total Path Cost",
      f"{total_cost:.2f} km" if solution_path else "No Path Found",
  )
  col3.metric("Nodes Expanded", len(expansion_seq))

  st.write(
      "**Expansion Sequence:**",
      " ➔ ".join(expansion_seq) if expansion_seq else "None",
  )
  st.write(
      "**Solution Path:**",
      " ➔ ".join(solution_path) if solution_path else "No Path Found",
  )