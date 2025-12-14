"""Script to bucket images into category folders on Desktop."""

import argparse
import shutil
import sys
from pathlib import Path

from image_bucketing.bucketer import ImageBucketer


def sanitize_folder_name(name: str) -> str:
    """Sanitize category name for use as folder name."""
    # Replace invalid filesystem characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()


def get_unique_filename(dest_dir: Path, filename: str) -> Path:
    """Get unique filename, adding _1, _2, etc. if needed."""
    dest_path = dest_dir / filename
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = dest_dir / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def main():
    parser = argparse.ArgumentParser(
        description="Bucket images into category folders on Desktop"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/con/ideas-syncthing/z-to-sort/images/",
        help="Input folder containing images (default: /Users/con/ideas-syncthing/z-to-sort/images/)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help='Category labels, e.g. --categories "cars" "screenshots" "people"',
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output folder on Desktop (default: ~/Desktop/images_bucketed/)",
    )
    parser.add_argument(
        "--save-index",
        type=str,
        help="Optional: save index to file for later search",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/clip-ViT-L-14",
        help="Model to use for embeddings. Default: clip-ViT-L-14 (best accuracy). "
             "Options: clip-ViT-B-16 (balanced), clip-ViT-B-32 (fastest)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        desktop = Path.home() / "Desktop"
        output_path = desktop / "images_bucketed"

    # Create output directory
    print(f"[1/4] Setting up output directory...", file=sys.stderr)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"      Output: {output_path}", file=sys.stderr)

    # Build index
    print(f"\n[2/4] Processing images from: {input_path}", file=sys.stderr)
    print(f"      Categories: {', '.join(args.categories)}", file=sys.stderr)
    print(f"      Model: {args.model}", file=sys.stderr)
    bucketer = ImageBucketer(model_name=args.model)
    index = bucketer.build_index(str(input_path), args.categories)

    if not index["items"]:
        print("No images found to process.", file=sys.stderr)
        sys.exit(0)

    total_images = len(index["items"])
    print(f"      Found {total_images} images to process", file=sys.stderr)

    # Create category folders
    print(f"\n[3/4] Creating category folders...", file=sys.stderr)
    category_folders = {}
    for cat in args.categories:
        safe_name = sanitize_folder_name(cat)
        cat_folder = output_path / safe_name
        cat_folder.mkdir(exist_ok=True)
        category_folders[cat] = cat_folder
        print(f"      Created: {cat_folder.name}/", file=sys.stderr)

    # Copy images to category folders
    print(f"\n[4/4] Copying images to category folders...", file=sys.stderr)
    copied = 0
    failed = 0
    for i, item in enumerate(index["items"], 1):
        src_path = Path(item["path"])
        dest_folder = category_folders[item["category"]]
        dest_path = get_unique_filename(dest_folder, src_path.name)

        try:
            shutil.copy2(src_path, dest_path)
            copied += 1
            if i % 10 == 0 or i == total_images:
                pct = (i / total_images) * 100
                print(f"      Progress: {i}/{total_images} ({pct:.1f}%) - {item['category']}", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"      Error copying {src_path.name}: {e}", file=sys.stderr)

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"✓ Completed: {copied} images copied", file=sys.stderr)
    if failed > 0:
        print(f"⚠ Failed: {failed} images", file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)
    
    summary = bucketer.get_bucket_summary(index)
    print(f"\n[Category distribution]", file=sys.stderr)
    for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
        pct = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  {cat:<20} {count:>5} images ({pct:>5.1f}%)", file=sys.stderr)

    # Save index if requested
    if args.save_index:
        print(f"\n[Saving index...]", file=sys.stderr)
        bucketer.save_index(index, args.save_index)
        print(f"✓ Index saved to: {args.save_index}", file=sys.stderr)


if __name__ == "__main__":
    main()

