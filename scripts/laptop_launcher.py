"""Start a self-contained local demonstration using the installed Python environment."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def health(url):
    try:
        with urlopen(url + "/api/health", timeout=3) as response:
            return json.load(response)
    except (OSError, URLError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    current = health(url)
    if current is not None and current.get("status") != "ok":
        raise RuntimeError(f"Port {args.port} has an unhealthy service: {current}")
    if current is None:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"Port {args.port} is occupied; use --port with another port")
        for relative in ["artifacts/siglip_adapted/image_embeddings.npy", "data/snapshots/collection_v2.jsonl", "web/dist/index.html"]:
            if not (ROOT / relative).is_file():
                raise FileNotFoundError(f"Missing runtime resource: {relative}")
        logs = ROOT / "logs/laptop"
        logs.mkdir(parents=True, exist_ok=True)
        runtime_file = logs / "runtime.json"
        runtime = json.loads(runtime_file.read_text(encoding="utf-8-sig")) if runtime_file.exists() else {}
        model_cache = runtime.get("model_cache", str(ROOT / "models/huggingface/hub"))
        env = os.environ.copy()
        env.update(
            CURIO_ENCODER="siglip", CURIO_INDEX="artifacts/siglip_adapted",
            CURIO_DATA="data/snapshots/collection_v2.jsonl", CURIO_PROBE_WEIGHT="0",
            CURIO_PROBES="artifacts/probes/siglip_adapted",
            CURIO_METRICS="artifacts/evaluation/image_audit/selected_test/metrics.json",
            HF_HUB_CACHE=model_cache,
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1",
        )
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (logs / "server.log").open("ab") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "curio.serve.app:app", "--host", "127.0.0.1", "--port", str(args.port)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                creationflags=flags,
            )
        (logs / "server.json").write_text(json.dumps({"pid": process.pid, "url": url, "root": str(ROOT)}), encoding="utf-8")
        for _ in range(90):
            if health(url):
                break
            if process.poll() is not None:
                raise RuntimeError(f"Server exited. See {logs / 'server.log'}")
            time.sleep(1)
        else:
            raise TimeoutError(f"Server startup timed out. See {logs / 'server.log'}")
    print(f"Curio ready: {url}")
    if not args.no_browser:
        webbrowser.open(url)


if __name__ == "__main__":
    main()
