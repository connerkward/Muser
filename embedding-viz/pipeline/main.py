"""
Main pipeline: embed -> cluster -> phylogeny
"""
import argparse
from embed import main as embed_main
from cluster import main as cluster_main
from phylogeny import main as phylogeny_main


def run_pipeline(mode: str = "image", source_dir: str = None, max_items: int = 500):
    print("=" * 50)
    print(f"STEP 1: Generating {mode} embeddings")
    print("=" * 50)
    embed_main(mode=mode, source_dir=source_dir, max_items=max_items)
    
    print("\n" + "=" * 50)
    print("STEP 2: UMAP + Clustering")
    print("=" * 50)
    cluster_main(mode=mode)
    
    print("\n" + "=" * 50)
    print("STEP 3: Building phylogeny tree")
    print("=" * 50)
    phylogeny_main(mode=mode)
    
    print("\n" + "=" * 50)
    print("Pipeline complete! Output: data/embeddings.json")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embedding visualization pipeline")
    parser.add_argument("--mode", choices=["image", "text"], default="image",
                        help="Embedding mode: image or text")
    parser.add_argument("--input", type=str, help="Source directory path")
    parser.add_argument("--max", type=int, default=500, help="Max items to process")
    
    args = parser.parse_args()
    run_pipeline(mode=args.mode, source_dir=args.input, max_items=args.max)
