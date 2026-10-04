"""Download acoustic CTC forced aligner model (Wav2Vec2 Arabic) into a local folder."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
ONNX_REPO_ID = "Alimalas/munajjam-onnx-models"
DEFAULT_DIR = Path("munajjam/models/model_wav2vec2")


def download(local_dir=DEFAULT_DIR, token=None, onnx=False) -> Path:
    if onnx:
        path = snapshot_download(
            repo_id=ONNX_REPO_ID,
            allow_patterns=["model_wav2vec2/*"],
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False,
        )
    else:
        path = snapshot_download(
            repo_id=REPO_ID, local_dir=str(local_dir), token=token
        )
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    parser.add_argument("--onnx", action="store_true", help="Download pre-compiled ONNX DirectML model from Alimalas/munajjam-onnx-models")
    args = parser.parse_args()
    print(f"Downloaded to: {download(args.local_dir, args.token, args.onnx)}")


if __name__ == "__main__":
    main()

