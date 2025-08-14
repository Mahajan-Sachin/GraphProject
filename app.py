import streamlit as st
import pandas as pd
import io
import networkx as nx
import matplotlib.pyplot as plt
from route_finder import Graph, load_graph_from_csv, dijkstra, effective_distance_km

st.set_page_config(page_title="Intelligent Route Finder", layout="wide")

st.title("🚦 Intelligent Route Finder & Traffic Estimator")
st.markdown("Built with Graphs, Dijkstra's algorithm, and Streamlit UI — Resume-ready project.")

# Sidebar: data input
st.sidebar.header("Graph Input")
input_mode = st.sidebar.radio("Input method:", ("Use sample graph", "Upload CSV", "Add manually"))

# Sample CSV content (small realistic graph)
sample_csv = """A,B,10,20
B,C,5,0
A,C,15,10
C,D,10,30
B,D,20,0
D,E,5,0
E,F,8,10
B,F,25,0
G,H,12,0
H,E,40,10
"""

if input_mode == "Use sample graph":
    st.sidebar.markdown("Using bundled sample graph with 9 edges.")
    g = load_graph_from_csv(io.StringIO(sample_csv)) if False else None
    # we need load_graph_from_csv to accept file-like — it doesn't; so parse manually here
    g = Graph()
    for line in sample_csv.strip().splitlines():
        src,dst,d,traffic = line.split(',')
        g.add_edge(src.strip(), dst.strip(), float(d.strip()), float(traffic.strip()))

elif input_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload edges CSV", type=['csv','txt'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, header=None)
            # expect 4 columns
            df = df.dropna(axis=0, how='all')
            g = Graph()
            for _, row in df.iterrows():
                try:
                    s = str(row[0]).strip(); t = str(row[1]).strip(); d = float(row[2]); tr = float(row[3])
                    g.add_edge(s,t,d,tr)
                except Exception:
                    continue
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            g = Graph()
    else:
        st.sidebar.info("Upload a CSV with columns: src,dst,distance_km,traffic_percent")
        g = Graph()

else:  # Add manually
    st.sidebar.markdown("Add edges manually (bidirectional by default)")
    manual_src = st.sidebar.text_input("Source node (e.g., A)")
    manual_dst = st.sidebar.text_input("Destination node (e.g., B)")
    manual_dist = st.sidebar.text_input("Distance in km (e.g., 10)")
    manual_traffic = st.sidebar.text_input("Traffic % (e.g., 20)")
    if 'manual_graph' not in st.session_state:
        st.session_state.manual_graph = Graph()
    if st.sidebar.button("Add edge"):
        try:
            s = manual_src.strip(); t = manual_dst.strip(); d = float(manual_dist); tr = float(manual_traffic)
            st.session_state.manual_graph.add_edge(s,t,d,tr)
            st.sidebar.success(f"Added edge {s} <-> {t} ({d} km, {tr}% traffic)")
        except Exception as e:
            st.sidebar.error("Please provide valid values before adding an edge.")
    g = st.session_state.manual_graph

# Main UI: show nodes & run query
cols = st.columns([2,1])
with cols[0]:
    st.subheader("Graph Summary")
    nodes = g.nodes()
    st.write(f"Nodes: {len(nodes)}")
    st.write(nodes)
    # show adjacency table
    rows = []
    for u in nodes:
        for (v, d, tr) in g.neighbors(u):
            rows.append({'src':u,'dst':v,'dist_km':d,'traffic_pct':tr})
    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("Graph is empty. Use sample graph or add edges.")

with cols[1]:
    st.subheader("Find Shortest Route")
    start = st.selectbox("Start node", options=nodes if nodes else [])
    target = st.selectbox("Destination node", options=nodes if nodes else [])
    speed_kmph = st.number_input("Average speed (km/h) — used for ETA", min_value=10.0, max_value=200.0, value=40.0)
    if st.button("Run Dijkstra"):
        if not start or not target:
            st.error("Please choose start and destination nodes")
        else:
            res = dijkstra(g, start, target)
            if res is None:
                st.error("No path found between nodes")
            else:
                eff_dist, path, base_dist = res
                eta_hours = eff_dist / speed_kmph
                eta_minutes = eta_hours * 60.0
                st.success(f"Path found: {' → '.join(path)}")
                st.write(f"Total base distance: {base_dist:.2f} km")
                st.write(f"Total distance (with traffic): {eff_dist:.2f} km")
                st.write(f"Estimated time at {speed_kmph} km/h: {int(eta_minutes)} minutes")

# Visualization
st.subheader("Route Visualization")
if nodes:
    try:
        G_nx = nx.Graph()
        for u in nodes:
            G_nx.add_node(u)
        for u in nodes:
            for (v, d, tr) in g.neighbors(u):
                if G_nx.has_edge(u,v):
                    continue
                G_nx.add_edge(u,v,weight=d,traffic=tr)

        fig, ax = plt.subplots(figsize=(8,5))
        pos = nx.spring_layout(G_nx, seed=42)
        nx.draw(G_nx, pos, with_labels=True, node_size=600, ax=ax)

        # highlight last computed path if any
        if 'path' in locals() and path:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G_nx, pos, edgelist=path_edges, width=3.0)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Visualization error: {e}")
else:
    st.info("No nodes to visualize")

st.markdown("---")
st.markdown("**Notes:** This app uses Dijkstra's algorithm where edge effective distance = base_distance * (1 + traffic_percent/100).\nYou can export the graph CSV and re-upload it to reproduce the demo.")


 