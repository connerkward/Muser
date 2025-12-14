"""
Phylogeny tree construction using similarity + temporal constraints.
Groups nodes into "species" based on cluster membership and time periods.
"""
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

DATA_DIR = Path(__file__).parent.parent / "data"

TEMPORAL_THRESHOLD = 30 * 24 * 60 * 60  # 30 days
SIMILARITY_THRESHOLD = 0.85


def load_clustered_data(mode: str = None) -> dict:
    """Load clustered embeddings."""
    if mode:
        path = DATA_DIR / f"embeddings_clustered_{mode}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    with open(DATA_DIR / "embeddings_clustered.json") as f:
        return json.load(f)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    similarity = embeddings @ embeddings.T
    return similarity


def build_constrained_graph(
    similarity: np.ndarray,
    timestamps: np.ndarray
) -> np.ndarray:
    """Build distance graph with temporal constraints."""
    n = len(timestamps)
    distances = np.full((n, n), np.inf)
    
    for i in range(n):
        for j in range(i + 1, n):
            time_diff = abs(timestamps[i] - timestamps[j])
            sim = similarity[i, j]
            
            temporally_close = time_diff < TEMPORAL_THRESHOLD
            highly_similar = sim > SIMILARITY_THRESHOLD
            
            if temporally_close or highly_similar:
                dist = 1 - sim
                dist += (time_diff / TEMPORAL_THRESHOLD) * 0.01
                distances[i, j] = dist
                distances[j, i] = dist
    
    return distances


def generate_species(items: list[dict], clusters: list[dict]) -> list[dict]:
    """
    Generate species by grouping clusters with time periods.
    Each species is a cluster + time range combination.
    """
    # Group items by cluster
    cluster_items = defaultdict(list)
    for item in items:
        if item["cluster"] >= 0:
            cluster_items[item["cluster"]].append(item)
    
    species = []
    cluster_map = {c["id"]: c for c in clusters}
    
    for cluster_id, items_in_cluster in cluster_items.items():
        if cluster_id not in cluster_map:
            continue
            
        cluster = cluster_map[cluster_id]
        
        # Get time range
        timestamps = [item["timestamp"] for item in items_in_cluster]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        
        # Format dates
        min_date = datetime.fromtimestamp(min_ts).strftime("%Y-%m")
        max_date = datetime.fromtimestamp(max_ts).strftime("%Y-%m")
        
        if min_date == max_date:
            date_range = min_date
        else:
            date_range = f"{min_date} → {max_date}"
        
        species.append({
            "id": f"species_{cluster_id}",
            "cluster_id": cluster_id,
            "name": cluster["label"],
            "date_range": date_range,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "centroid": cluster["centroid"],
            "count": len(items_in_cluster),
            "item_ids": [item["id"] for item in items_in_cluster]
        })
    
    # Sort by earliest timestamp
    species.sort(key=lambda s: s["min_timestamp"])
    
    return species


def build_phylogeny_tree(distances: np.ndarray, items: list[dict]) -> dict:
    """Build MST and convert to hierarchical structure."""
    print("Building minimum spanning tree...")
    
    distances_fallback = distances.copy()
    distances_fallback[distances_fallback == np.inf] = 1000
    
    mst = minimum_spanning_tree(distances_fallback)
    mst_array = mst.toarray()
    
    # Find root (earliest timestamp)
    timestamps = [item["timestamp"] for item in items]
    root_idx = np.argmin(timestamps)
    root_ts = min(timestamps)
    
    # Format root date
    root_date = datetime.fromtimestamp(root_ts).strftime("%Y-%m-%d")
    
    # BFS from root
    n = len(items)
    visited = [False] * n
    parent = [-1] * n
    queue = [root_idx]
    visited[root_idx] = True
    
    mst_symmetric = mst_array + mst_array.T
    
    while queue:
        node = queue.pop(0)
        for neighbor in range(n):
            if mst_symmetric[node, neighbor] > 0 and not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(neighbor)
    
    for i in range(n):
        if not visited[i]:
            parent[i] = root_idx
    
    # Build nodes list
    nodes = []
    for i, item in enumerate(items):
        ts = item["timestamp"]
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        node = {
            "id": item["id"],
            "parent": items[parent[i]]["id"] if parent[i] >= 0 else None,
            "timestamp": ts,
            "date": date_str,
            "cluster": item["cluster"]
        }
        nodes.append(node)
    
    print(f"Built tree with {len(nodes)} nodes, root: {items[root_idx]['id']} ({root_date})")
    
    return {
        "nodes": nodes,
        "root_id": items[root_idx]["id"],
        "root_date": root_date,
        "date_range": {
            "min": datetime.fromtimestamp(min(timestamps)).strftime("%Y-%m-%d"),
            "max": datetime.fromtimestamp(max(timestamps)).strftime("%Y-%m-%d")
        }
    }


def main(mode: str = None):
    data = load_clustered_data(mode)
    items = data["items"]
    clusters = data["clusters"]
    mode = data.get("mode", "image")
    
    # Extract embeddings and timestamps
    embeddings = np.array([item["embedding"] for item in items])
    timestamps = np.array([item["timestamp"] for item in items])
    
    print("Computing similarity matrix...")
    similarity = compute_similarity_matrix(embeddings)
    
    print("Building constrained graph...")
    distances = build_constrained_graph(similarity, timestamps)
    
    # Build phylogeny
    phylogeny = build_phylogeny_tree(distances, items)
    
    # Generate species
    print("Generating species...")
    species = generate_species(items, clusters)
    for s in species:
        print(f"  {s['name']}: {s['count']} items ({s['date_range']})")
    
    # Add to data
    data["phylogeny"] = phylogeny
    data["species"] = species
    
    # Remove raw embeddings
    for item in data["items"]:
        del item["embedding"]
    
    # Save final output
    output_path = DATA_DIR / f"embeddings_{mode}.json"
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    print(f"Saved final data to {output_path}")


if __name__ == "__main__":
    main()
