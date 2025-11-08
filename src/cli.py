import argparse
import sys
from src.core import distance, rearrange, utils


def main():
    parser = argparse.ArgumentParser(description="Sydnify — rearrange pixels like a pro.")
    parser.add_argument("--source", required=False, help="Path to source image")
    parser.add_argument("--target", required=False, help="Path to target image")
    parser.add_argument("--mode", choices=["heuristic", "optimal"], default="heuristic", help="Algorithm mode")
    args = parser.parse_args()

    print("✅ Sydnify initialized successfully!")
    print(f"Mode: {args.mode}")
    if args.source and args.target:
        print(f"Source: {args.source}")
        print(f"Target: {args.target}")
    else:
        print("No source/target paths provided yet — this is a test run.")
    sys.exit(0)

if __name__ == "__main__":
    main()
