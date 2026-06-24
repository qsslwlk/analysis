"""Command line entrypoint for the V2 observatory wrapper."""

from __future__ import annotations

import argparse

from observatoire.config import load_video_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 observatoire YouTube.")
    parser.add_argument("--config", default=None, help="JSON/YAML corpus config. Falls back to built-in POC videos.")
    parser.add_argument("--api-key", default=None, help="YouTube Data API v3 key. Otherwise YOUTUBE_API_KEY.")
    parser.add_argument("--max-comments-per-video", type=int, default=500)
    parser.add_argument("--include-replies", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--min-cluster-chars", type=int, default=80)
    parser.add_argument("--min-cluster-meaningful-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import observatoire_youtube_poc as poc

    videos = load_video_config(args.config) if args.config else poc.VIDEOS
    result = poc.run_pipeline(
        videos=videos,
        api_key=args.api_key,
        max_comments_per_video=args.max_comments_per_video,
        include_replies=args.include_replies,
        force_refresh=args.force_refresh,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        embedding_model=args.embedding_model,
        skip_embeddings=args.skip_embeddings,
        min_cluster_chars=args.min_cluster_chars,
        min_cluster_meaningful_tokens=args.min_cluster_meaningful_tokens,
    )
    print("\nExports générés :")
    for key in ["dashboard_path", "interpretation_note_path"]:
        print(f" - {result[key]}")


if __name__ == "__main__":
    main()
