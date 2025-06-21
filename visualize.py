import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pickle
import numpy as np
import imageio
import networkx as nx
import matplotlib.pyplot as plt
from run_neat import Genome, NodeGene, ConnectionGene
from evojax.task.slimevolley import SlimeVolley


# Set your genome and gif path here
GENOME_PATH = "models/slime_1038.pkl"
GIF_PATH = "results/slime_1038.gif"


def load_genome(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_and_record_slimevolley(genome, gif_path, max_steps=3000):
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    env = SlimeVolley(max_steps=max_steps, test=True)
    import jax

    key = jax.random.PRNGKey(0)
    state = env.reset(jax.random.split(key, 1))
    done = False
    frames = []
    t = 0
    # Score tracking
    score_agent = 0
    score_opponent = 0
    # Detect NEAT-Python genome
    is_neat = hasattr(genome, "nodes") and isinstance(genome.nodes, dict)
    if is_neat:
        import neat

        CONFIG_PATH = "slimevolley.ini"
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            CONFIG_PATH,
        )
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    while not done and t < max_steps:
        obs = np.array(state.obs[0])
        if is_neat:
            action_raw = net.activate(obs)
        else:
            action_raw = genome.feed_forward(obs)
        output = 1 / (1 + np.exp(-np.array(action_raw)))
        action = (output > 0.5).astype(int)
        import jax.numpy as jnp

        next_state, reward, terminated = env.step(state, jnp.expand_dims(action, 0))
        # Track scores (SlimeVolley: reward[0] is agent, reward[1] is opponent)
        if hasattr(reward, "__len__") and len(reward) == 2:
            score_agent += reward[0]
            score_opponent += reward[1]

        # Convert all fields in state to numpy scalars/arrays for rendering, but keep as State object
        def deep_to_numpy_state(state):
            import jax
            import jax.numpy as jnp
            import numpy as np

            if hasattr(state, "__dataclass_fields__"):
                return state.__class__(
                    **{
                        k: deep_to_numpy(getattr(state, k))
                        for k in state.__dataclass_fields__
                    }
                )
            return state

        def deep_to_numpy(obj):
            import jax
            import jax.numpy as jnp
            import numpy as np

            if isinstance(obj, (jax.Array, jnp.ndarray)):
                arr = np.asarray(obj)
                if arr.shape == ():
                    return arr.item()
                return arr
            elif isinstance(obj, dict):
                return {k: deep_to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(deep_to_numpy(v) for v in obj)
            elif hasattr(obj, "__dataclass_fields__"):
                return obj.__class__(
                    **{
                        k: deep_to_numpy(getattr(obj, k))
                        for k in obj.__dataclass_fields__
                    }
                )
            else:
                return obj

        render_state = deep_to_numpy_state(state)
        frame = np.asarray(SlimeVolley.render(render_state))
        frames.append(frame)
        done = bool(terminated[0])
        state = next_state
        t += 1
    # Add summary frame with score and winner
    if frames:
        import cv2

        frame = frames[-1].copy()
        h, w = frame.shape[:2]
        summary = f"Agent: {int(score_agent)}  Opponent: {int(score_opponent)}"
        if score_agent > score_opponent:
            winner = "Agent wins!"
        elif score_agent < score_opponent:
            winner = "Opponent wins!"
        else:
            winner = "Draw!"
        summary2 = f"{winner}"
        # Draw text on the frame (requires cv2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        color = (255, 255, 255)
        thickness = 3
        y0 = int(h * 0.1)
        cv2.putText(
            frame, summary, (40, y0), font, font_scale, color, thickness, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            summary2,
            (40, y0 + 50),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        frames.append(frame)
        imageio.mimsave(gif_path, frames, duration=0.005, loop=0)


def visualize_topology(genome, save_path):
    G = nx.DiGraph()
    # Support both NEAT-Python and evojax genome formats
    if isinstance(genome.nodes, dict):
        node_items = genome.nodes.items()
        node_ids = set(genome.nodes.keys())
        for in_node, out_node in genome.connections.keys():
            node_ids.add(in_node)
            node_ids.add(out_node)
        # Detect output nodes by type if possible
        output_nodes = [
            nid
            for nid, n in genome.nodes.items()
            if getattr(n, "type", None) == "output"
        ]
    else:
        node_items = [(n.id, n) for n in genome.nodes]
        node_ids = set(n.id for n in genome.nodes)
        for c in genome.connections:
            node_ids.add(c.in_node)
            node_ids.add(c.out_node)
        output_nodes = [
            n.id for n in genome.nodes if getattr(n, "type", None) == "output"
        ]
    # Fallback: if no output nodes detected by type, use legacy IDs (0,1,2,3)
    if not output_nodes:
        output_nodes = [nid for nid in node_ids if nid in [0, 1, 2, 3]]
    input_nodes = [nid for nid in node_ids if nid < 0]
    hidden_nodes = [
        nid for nid in node_ids if nid not in input_nodes and nid not in output_nodes
    ]

    # Build edge list
    if isinstance(genome.connections, dict):
        conn_items = genome.connections.items()
        enabled_edges = [
            (in_node, out_node)
            for (in_node, out_node), conn in conn_items
            if getattr(conn, "enabled", True)
        ]
    else:
        enabled_edges = [
            (c.in_node, c.out_node) for c in genome.connections if c.enabled
        ]
    # Compute incoming and outgoing for each node
    incoming = {nid: 0 for nid in node_ids}
    outgoing = {nid: 0 for nid in node_ids}
    for u, v in enabled_edges:
        outgoing[u] = outgoing.get(u, 0) + 1
        incoming[v] = incoming.get(v, 0) + 1
    # Only keep hidden nodes with at least 2 total connections and at least one outgoing (not a dead end)
    fully_connected_hidden = [
        nid
        for nid in hidden_nodes
        if (incoming.get(nid, 0) + outgoing.get(nid, 0)) >= 2
        and outgoing.get(nid, 0) > 0
    ]
    connected_outputs = [
        nid
        for nid in output_nodes
        if incoming.get(nid, 0) > 0 or outgoing.get(nid, 0) > 0
    ]
    # Final node set: all input, only connected outputs, and only fully connected hidden
    final_node_ids = set(input_nodes + connected_outputs + fully_connected_hidden)

    # Add all nodes to the graph
    activations = set()
    node_activations = {}
    for node_id in final_node_ids:
        activation = ""
        if isinstance(genome.nodes, dict) and node_id in genome.nodes:
            activation = getattr(genome.nodes[node_id], "activation", "")
        node_activations[node_id] = activation
        if node_id not in input_nodes:
            activations.add(activation)
        if node_id in input_nodes:
            ntype = "input"
        elif node_id in output_nodes:
            ntype = "output"
        else:
            ntype = "hidden"
        G.add_node(node_id, type=ntype, activation=activation)
    # Add edges (only between kept nodes)
    for u, v in enabled_edges:
        if u in final_node_ids and v in final_node_ids:
            # Find weight
            if isinstance(genome.connections, dict):
                conn = genome.connections.get((u, v))
            else:
                conn = next(
                    (
                        c
                        for c in genome.connections
                        if c.in_node == u and c.out_node == v
                    ),
                    None,
                )
            if conn is not None:
                G.add_edge(u, v, weight=conn.weight)
    # Node layout
    n_inputs = len([nid for nid in input_nodes if nid in final_node_ids])
    n_outputs = len([nid for nid in output_nodes if nid in final_node_ids])
    n_hidden = len(fully_connected_hidden)
    try:
        topo_order = list(nx.topological_sort(G))
    except Exception:
        topo_order = list(final_node_ids)
    node_depth = {nid: 0 for nid in final_node_ids}
    for nid in topo_order:
        preds = list(G.predecessors(nid))
        if preds:
            node_depth[nid] = max(node_depth[p] + 1 for p in preds)
        else:
            node_depth[nid] = 0
    max_depth = max(node_depth.values()) if node_depth else 1
    # Assign x positions: input=0, output=1, hidden=spread by depth
    pos = {}
    for i, nid in enumerate(sorted(input_nodes)):
        if nid in final_node_ids:
            pos[nid] = (0, 1 - 2 * i / max(n_inputs - 1, 1))
    for i, nid in enumerate(sorted(output_nodes)):
        if nid in final_node_ids:
            pos[nid] = (1, 1 - 2 * i / max(n_outputs - 1, 1))
    # Hidden nodes: spread by depth (x) and vertically
    hidden_by_depth = {}
    for nid in fully_connected_hidden:
        d = node_depth[nid]
        hidden_by_depth.setdefault(d, []).append(nid)
    for d, nids in sorted(hidden_by_depth.items()):
        for j, nid in enumerate(sorted(nids)):
            y = 1 - 2 * j / max(len(nids) - 1, 1) if len(nids) > 1 else 0
            x = 0.2 + 0.6 * (d / max_depth)  # spread hidden nodes between input/output
            pos[nid] = (x, y)
    for nid in final_node_ids:
        if nid not in pos:
            pos[nid] = (0.5, 0)
    color_map = {"input": "#4F81BD", "output": "#F79646", "hidden": "#9BBB59"}
    shape_map = {"input": "s", "output": "o", "hidden": "^"}
    plt.figure(figsize=(16, 8))
    # Assign a color to each activation function
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
    # Draw input nodes (always blue)
    if input_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[nid for nid in input_nodes if nid in final_node_ids],
            node_color="#4F81BD",
            node_shape="s",
            node_size=400,
            alpha=0.95,
        )
    # Draw output and hidden nodes by activation color
    for act in activations:
        act_nodes = [
            nid
            for nid in final_node_ids
            if node_activations[nid] == act and nid not in input_nodes
        ]
        if act_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=act_nodes,
                node_color=activation_colors[act],
                node_shape="o",
                node_size=400,
                alpha=0.95,
            )
    valid_edges = [(u, v) for u, v in G.edges() if u in pos and v in pos]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=valid_edges,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.5,
    )
    # Edge labels
    if isinstance(genome.connections, dict):
        edge_labels = {
            (in_node, out_node): f"{conn.weight:.2f}"
            for (in_node, out_node), conn in genome.connections.items()
            if getattr(conn, "enabled", True)
            and in_node in final_node_ids
            and out_node in final_node_ids
        }
    else:
        edge_labels = {
            (c.in_node, c.out_node): f"{c.weight:.2f}"
            for c in genome.connections
            if c.enabled
            and c.in_node in final_node_ids
            and c.out_node in final_node_ids
        }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    # Node labels: input nodes show 'input', output nodes show activation, hidden nodes show activation only
    node_labels = {}
    for nid in final_node_ids:
        if nid in input_nodes:
            node_labels[nid] = "input"
        elif nid in output_nodes:
            activation = node_activations[nid]
            node_labels[nid] = f"output\n{activation}" if activation else "output"
        else:
            activation = node_activations[nid]
            node_labels[nid] = f"{activation}" if activation else ""
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    # Legend for activation functions
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
    plt.legend(handles=legend_elements, loc="upper left")
    plt.title("NEAT Genome Topology (Input: left, Output: right)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_network(genome):
    print("\n=== NEAT Genome Network ===")
    print(genome)
    for node in genome.nodes:
        print(genome.nodes[node])
    for node in sorted(genome.nodes, key=lambda n: n.id):
        if node.type == "input":
            print(f"Node {node.id}: INPUT")
        else:
            print(f"Node {node.id}: {node.type.upper()} ({node.activation})")
    print("\nConnections (enabled only):")
    for conn in genome.connections:
        if conn.enabled:
            print(
                f"  {conn.in_node} -> {conn.out_node} | w={conn.weight:.3f} | innov={conn.innovation}"
            )
    print("==========================\n")


if __name__ == "__main__":
    genome = load_genome(GENOME_PATH)
    # print_network(genome)
    # run_and_record_slimevolley(genome, GIF_PATH)
    topo_path = GIF_PATH.replace(".gif", "_topology.png")
    visualize_topology(genome, topo_path)
    print(f"[INFO] Topology image saved to {topo_path}")
