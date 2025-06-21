import sys
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from neat.genes import DefaultNodeGene


def load_genome(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_genome(genome):
    """Clean the genome by removing disconnected and dead-end nodes."""
    print("[DEBUG] Starting genome cleaning")
    print(f"Initial nodes: {len(genome.nodes)}")
    print(f"Initial connections: {len(genome.connections)}")

    # First get all enabled connections
    enabled_conns = {
        (in_node, out_node): conn
        for (in_node, out_node), conn in genome.connections.items()
        if conn.enabled
    }
    print(f"Enabled connections: {len(enabled_conns)}")

    # Build connection maps
    in_edges = {}
    out_edges = {}
    all_nodes = set()

    for (in_node, out_node), _ in enabled_conns.items():
        in_edges.setdefault(out_node, set()).add(in_node)
        out_edges.setdefault(in_node, set()).add(out_node)
        all_nodes.add(in_node)
        all_nodes.add(out_node)

    # Identify node types (including nodes that might not be in genome.nodes)
    input_nodes = {k for k in all_nodes if k < 0}
    output_nodes = {0, 1, 2}  # Fixed output nodes
    hidden_nodes = {k for k in all_nodes if k >= 0 and k not in output_nodes}

    print(f"Input nodes: {sorted(input_nodes)}")
    print(f"Output nodes: {sorted(output_nodes)}")
    print(f"Hidden nodes: {len(hidden_nodes)}")

    # Start with output nodes that have incoming connections
    connected_outputs = {node for node in output_nodes if node in in_edges}
    valid_nodes = set(connected_outputs)
    print(f"Connected outputs: {sorted(connected_outputs)}")

    if connected_outputs:
        # Work backwards from outputs to inputs
        to_process = list(connected_outputs)
        while to_process:
            node = to_process.pop(0)
            if node in in_edges:
                for pred in in_edges[node]:
                    if pred not in valid_nodes:
                        valid_nodes.add(pred)
                        to_process.append(pred)

    print(f"Valid nodes found: {sorted(valid_nodes)}")

    # For each valid node, ensure it exists in genome.nodes
    for node_id in valid_nodes:
        if node_id not in genome.nodes:
            # Create a new node with default values
            genome.nodes[node_id] = DefaultNodeGene(node_id)

    # Keep only valid nodes and their connections
    genome.nodes = {k: v for k, v in genome.nodes.items() if k in valid_nodes}
    genome.connections = {
        k: v
        for k, v in enabled_conns.items()
        if k[0] in valid_nodes and k[1] in valid_nodes
    }

    print(f"Final nodes: {len(genome.nodes)}")
    print(f"Final connections: {len(genome.connections)}")
    print("Nodes kept:", sorted(genome.nodes.keys()))
    print("Connections kept:", sorted(genome.connections.keys()))

    return genome


def visualize_topology(genome, save_path):
    """Visualize the genome topology."""
    print("[DEBUG] Starting visualization")

    G = nx.DiGraph()

    # Input/output node names
    input_names = {
        -1: "x",
        -2: "y",
        -3: "vx",
        -4: "vy",
        -5: "ball_x",
        -6: "ball_y",
        -7: "ball_vx",
        -8: "ball_vy",
        -9: "op_x",
        -10: "op_y",
        -11: "op_vx",
        -12: "op_vy",
    }

    output_names = {0: "forward", 1: "backward", 2: "jump"}

    # Identify node types
    input_nodes = {k for k in genome.nodes.keys() if k < 0}
    output_nodes = {0, 1, 2}  # Fixed output nodes
    hidden_nodes = {k for k in genome.nodes.keys() if k >= 0 and k not in output_nodes}

    print(
        f"Nodes to draw - inputs: {sorted(input_nodes)}, outputs: {sorted(output_nodes & genome.nodes.keys())}, hidden: {len(hidden_nodes)}"
    )

    # Add nodes to graph and collect activation types
    activations = set()
    node_activations = {}

    for node_id, node in genome.nodes.items():
        activation = getattr(node, "activation", "")
        node_activations[node_id] = activation
        if node_id not in input_nodes:  # Only track non-input activations for legend
            activations.add(activation)

        if node_id in input_nodes:
            ntype = "input"
        elif node_id in output_nodes:
            ntype = "output"
        else:
            ntype = "hidden"
        G.add_node(node_id, type=ntype, activation=activation)

    # Add edges
    for (in_node, out_node), conn in genome.connections.items():
        if in_node in genome.nodes and out_node in genome.nodes:
            G.add_edge(in_node, out_node, weight=conn.weight)

    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Node layout
    try:
        topo_order = list(nx.topological_sort(G))
    except Exception:
        topo_order = list(G.nodes())

    # Calculate node depths
    node_depth = {nid: 0 for nid in G.nodes()}
    for nid in topo_order:
        preds = list(G.predecessors(nid))
        if preds:
            node_depth[nid] = max(node_depth[p] + 1 for p in preds)

    max_depth = max(node_depth.values()) if node_depth else 1

    # Position nodes
    pos = {}
    n_inputs = len(input_nodes)
    n_outputs = len(output_nodes & set(G.nodes()))

    # Input nodes on the left
    for i, nid in enumerate(sorted(input_nodes)):
        pos[nid] = (0, 1 - 2 * i / max(n_inputs - 1, 1))

    # Output nodes on the right
    for i, nid in enumerate(sorted(output_nodes & set(G.nodes()))):
        pos[nid] = (1, 1 - 2 * i / max(n_outputs - 1, 1))

    # Hidden nodes by depth
    hidden_by_depth = {}
    for nid in hidden_nodes:
        d = node_depth[nid]
        hidden_by_depth.setdefault(d, []).append(nid)

    for d, nids in sorted(hidden_by_depth.items()):
        for j, nid in enumerate(sorted(nids)):
            y = 1 - 2 * j / max(len(nids) - 1, 1) if len(nids) > 1 else 0
            x = 0.2 + 0.6 * (d / max_depth)  # spread between input/output
            pos[nid] = (x, y)

    # Default position for any remaining nodes
    for nid in G.nodes():
        if nid not in pos:
            pos[nid] = (0.5, 0)

    # Set up colors
    activation_colors = {}
    color_palette = [
        "#9BBB59",
        "#C0504D",
        "#4BACC6",
        "#8064A2",
        "#F79646",
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
    ]

    for i, act in enumerate(sorted(activations)):
        activation_colors[act] = color_palette[i % len(color_palette)]

    # Increased node sizes
    input_node_size = 1000
    hidden_node_size = 800
    output_node_size = 1000

    # Draw the network
    plt.figure(figsize=(9, 5))  # Increased figure size

    # Draw input nodes
    if input_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(input_nodes),
            node_color="#4F81BD",
            node_shape="s",
            node_size=input_node_size,
            alpha=0.95,
        )

    # Draw other nodes by activation
    for act in activations:
        act_nodes = [
            nid
            for nid in G.nodes()
            if node_activations[nid] == act and nid not in input_nodes
        ]
        if act_nodes:
            is_output = any(n in output_nodes for n in act_nodes)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=act_nodes,
                node_color=activation_colors[act],
                node_shape="o",
                node_size=output_node_size if is_output else hidden_node_size,
                alpha=0.95,
            )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,  # Increased arrow size
        width=1.5,
    )

    # Edge labels
    edge_labels = {(u, v): f"{G.edges[u, v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Node labels using descriptive names
    node_labels = {}
    for nid in G.nodes():
        if nid in input_nodes:
            node_labels[nid] = input_names.get(nid, f"in_{abs(nid)}")
        elif nid in output_nodes:
            activation = node_activations[nid]
            name = output_names.get(nid, f"out_{nid}")
            node_labels[nid] = f"{name}\n{activation}" if activation else name
        else:
            activation = node_activations[nid]
            node_labels[nid] = f"{activation}" if activation else ""

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor="#4F81BD", edgecolor="k", label="input")]
    for act in sorted(activations):
        legend_elements.append(
            Patch(
                facecolor=activation_colors[act],
                edgecolor="k",
                label=act if act else "(none)",
            )
        )

    # Place legend at the bottom center, outside the plot
    plt.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(legend_elements),
        frameon=False,
    )
    plt.title("NEAT Genome Topology (Input: left, Output: right)")
    plt.axis("off")
    # Add padding at the bottom for the legend
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(save_path, dpi=300)  # Increased DPI for better quality
    plt.close()


GENOME_PATH = ()
if __name__ == "__main__":
    genome_path = "models/slime_1038.pkl"
    save_path = "graphs/slime_1038_topology.png"

    genome = load_genome(genome_path)
    genome = clean_genome(genome)  # Clean before visualizing
    visualize_topology(genome, save_path)
    print(f"[INFO] Topology image saved to {save_path}")
