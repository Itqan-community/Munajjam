"""Download obadx/recitation-segmenter-v2 into a local folder."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "obadx/recitation-segmenter-v2"
DEFAULT_DIR = Path("munajjam/models/recitation-segmenter-v2")


def download(local_dir=DEFAULT_DIR, token=None) -> Path:
    path = snapshot_download(
        repo_id=REPO_ID, local_dir=str(local_dir), token=token
    )
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()
    print(f"Downloaded to: {download(args.local_dir, args.token)}")


if __name__ == "__main__":
    main()
