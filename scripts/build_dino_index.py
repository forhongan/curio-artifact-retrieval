"""Build a separate frozen DINOv2 reference-image index."""

import json

from curio.index.build import build_index
from curio.represent.image_encoder import DinoEncoder


if __name__ == "__main__":
    encoder = DinoEncoder()
    print(json.dumps(build_index(output="artifacts/dinov2", encoder_name="dinov2", encoder=encoder), indent=2))
