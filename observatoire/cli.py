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
    parser.add_argument("--extract-discourse-cards", action="store_true")
    parser.add_argument("--cluster-discourse-cards", action="store_true")
    parser.add_argument("--extract-claims", action="store_true")
    parser.add_argument("--cluster-claims", action="store_true")
    parser.add_argument("--label-claim-clusters", action="store_true")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--max-claims-per-comment", type=int, default=3)
    parser.add_argument("--claim-min-confidence", type=float, default=0.65)
    parser.add_argument("--claim-extraction-limit", type=int, default=None)
    parser.add_argument("--claim-cluster-min-size", type=int, default=8)
    parser.add_argument("--claim-cluster-distance-threshold", type=float, default=0.35)
    parser.add_argument("--discursive-card-min-confidence", type=float, default=0.55)
    parser.add_argument("--discursive-card-limit", type=int, default=None)
    parser.add_argument("--discursive-cluster-min-size", type=int, default=6)
    parser.add_argument("--discursive-cluster-distance-threshold", type=float, default=0.35)
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
        extract_discourse_cards=args.extract_discourse_cards,
        cluster_discourse_cards=args.cluster_discourse_cards,
        extract_claims=args.extract_claims,
        cluster_claims=args.cluster_claims,
        label_claim_clusters=args.label_claim_clusters,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        max_claims_per_comment=args.max_claims_per_comment,
        claim_min_confidence=args.claim_min_confidence,
        claim_extraction_limit=args.claim_extraction_limit,
        claim_cluster_min_size=args.claim_cluster_min_size,
        claim_cluster_distance_threshold=args.claim_cluster_distance_threshold,
        discursive_card_min_confidence=args.discursive_card_min_confidence,
        discursive_card_limit=args.discursive_card_limit,
        discursive_cluster_min_size=args.discursive_cluster_min_size,
        discursive_cluster_distance_threshold=args.discursive_cluster_distance_threshold,
    )
    print("\nExports générés :")
    for key in ["dashboard_path", "interpretation_note_path"]:
        print(f" - {result[key]}")


if __name__ == "__main__":
    main()
