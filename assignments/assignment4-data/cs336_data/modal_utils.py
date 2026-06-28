from pathlib import Path, PurePosixPath

import modal

from cs336_data.common import MODAL_SHARED_PATH

SUNET_ID = "00000001"  # Self-study placeholder (replace with 8-digit Stanford ID if enrolled)

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"data-{SUNET_ID}")
data_volume = modal.Volume.from_name(f"data-{SUNET_ID}", create_if_missing=True, version=2)
shared_data_volume = modal.Volume.from_name(
    "a4-shared-data", create_if_missing=True, version=2
)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.12")
    image = image.uv_sync()
    image = image.add_local_python_source("cs336_basics")
    image = image.add_local_python_source("cs336_data")
    image = image.add_local_file("AGENTS.md", "/root/AGENTS.md")
    image = image.add_local_file("CLAUDE.md", "/root/CLAUDE.md")
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/root/data": data_volume,
    str(MODAL_SHARED_PATH): shared_data_volume.read_only(),
}

MODAL_SECRETS = []
