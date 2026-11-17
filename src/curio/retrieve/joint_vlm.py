"""Query-conditioned VQA scoring via a persistent, isolated local worker."""

import atexit
import hashlib
import json
import math
import re
from pathlib import Path
import selectors
import subprocess
import threading

from .joint_pixels import color_observability
from .joint_query import plan_free_query

VERSION = "joint-vqa-v1-448"


def validate_score(result):
    """Malformed/NaN inference is a failed job, never a negative decision."""
    probability = result.get("score")
    if (
        not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise RuntimeError("视觉模型返回了无效分数，本次任务未生成完整结果。")
    if result.get("reason") != "color_not_observable":
        margin = result.get("yes_no_logit")
        mass = result.get("answer_mass")
        if not isinstance(margin, (int, float)) or not math.isfinite(margin):
            raise RuntimeError("视觉模型返回了无效判别值。")
        if (
            not isinstance(mass, (int, float))
            or not math.isfinite(mass)
            or not 0 <= mass <= 1.00001
        ):
            raise RuntimeError("视觉模型返回了无效回答分布。")
    return result


class LocalVQA:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root / "artifacts/joint_vlm_v9"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._process = None
        self._stderr = None
        self._lock = threading.RLock()
        self._plans = {}
        self._identity = None
        atexit.register(self.close)

    @property
    def identity(self):
        if self._identity is None:
            receipt_path = self.directory / "model_receipt.json"
            if not receipt_path.exists():
                raise RuntimeError("尚未安装本地视觉模型，请运行 scripts/setup_joint_vlm.sh。")
            receipt = json.loads(receipt_path.read_text())
            self._identity = {
                "version": VERSION,
                "repository": receipt["repository"],
                "revision": receipt["revision"],
                "worker_sha256": hashlib.sha256(
                    (self.root / "scripts/joint_vlm_worker.py").read_bytes()
                ).hexdigest(),
            }
        return self._identity

    def _read(self, timeout=180):
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                self.close()
                raise RuntimeError("视觉模型响应超时；本次任务失败，未生成全库否定结果。")
        line = self._process.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("视觉模型进程退出，请检查 artifacts/joint_vlm_v9/worker.log。")
        try:
            return json.loads(line)
        except ValueError as error:
            self.close()
            raise RuntimeError("视觉模型通信格式错误。") from error

    def _start(self):
        if self._process and self._process.poll() is None:
            return
        self.identity  # validates installation before launching
        python = self.root / ".venv-vlm/bin/python"
        model = self.root / "models/vlm_qwen3_8b_4bit"
        if not python.exists() or not (model / "config.json").exists():
            raise RuntimeError("尚未安装本地视觉模型，请运行 scripts/setup_joint_vlm.sh。")
        self._stderr = (self.directory / "worker.log").open("a")
        self._process = subprocess.Popen(
            [
                str(python),
                "-u",
                str(self.root / "scripts/joint_vlm_worker.py"),
                "--root",
                str(self.root),
                "--model",
                str(model),
            ],
            cwd=self.root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        if not self._read().get("ready"):
            self.close()
            raise RuntimeError("视觉模型未能完成加载。")

    def _request(self, payload):
        with self._lock:
            self._start()
            try:
                self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                value = self._read()
            except (OSError, BrokenPipeError) as error:
                self.close()
                raise RuntimeError("视觉模型连接中断。") from error
            if not value.get("ok"):
                raise RuntimeError("视觉模型判别失败：" + str(value.get("error", "unknown")))
            return value

    def plan(self, query):
        with self._lock:
            if query not in self._plans:
                self._plans[query] = plan_free_query(query, self._translate)
            return self._plans[query]

    def _translate(self, text):
        self.identity
        # Free queries are planned before scoring. Release our scoring model to
        # avoid retaining two copies of an 8B model in the same service.
        self.close()
        try:
            with (self.directory / "planner.log").open("a") as log:
                result = subprocess.run(
                    [
                        str(self.root / ".venv-vlm/bin/python"),
                        str(self.root / "scripts/joint_query_worker.py"),
                        "--model",
                        str(self.root / "models/vlm_qwen3_8b_4bit"),
                    ],
                    input=json.dumps({"query": text}, ensure_ascii=False),
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    timeout=180,
                    check=True,
                )
            plan = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise RuntimeError("查询翻译失败，未生成全库否定结果；请检查 planner.log。") from error
        if not re.search(r"[\u3400-\u9fff]", text):
            # Preserve English constraints verbatim, even if the translator adds
            # a definition despite its instructions.
            plan["canonical"] = text.strip()
        return plan

    def score(self, image_path, query, *, references=()):
        path = (self.root / image_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Image must be inside the project")
        stat = path.stat()
        payload = {
            "model": self.identity,
            "question": query.question,
            "requires_color": query.requires_color,
            "path": str(path.relative_to(self.root)),
            "size": stat.st_size,
            "mtime": stat.st_mtime_ns,
            "references": list(references),
        }
        if references:
            stamps = []
            for reference in references:
                ref_path = (self.root / reference["image_path"]).resolve()
                if not ref_path.is_relative_to(self.root) or not ref_path.is_file():
                    raise ValueError("Reference image must be inside the project")
                ref_stat = ref_path.stat()
                stamps.append((str(ref_path), ref_stat.st_size, ref_stat.st_mtime_ns))
            payload["reference_stamps"] = stamps
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache = self.directory / "scores" / key[:2] / (key + ".json")
        with self._lock:
            if cache.exists():
                return {**validate_score(json.loads(cache.read_text())), "cached": True}
            observable = color_observability(path) if query.requires_color else None
            if observable is not None and not observable["observable"]:
                result = {
                    "score": 0.0,
                    "yes_no_logit": None,
                    "answer_mass": None,
                    "reason": "color_not_observable",
                    "color_observability": observable,
                    "latency_ms": 0.0,
                }
            else:
                result = self._request(
                    {
                        "operation": "score",
                        "image_path": payload["path"],
                        "question": query.question,
                        "references": list(references),
                    }
                )
                result.update(reason="model_score", color_observability=observable)
            validate_score(result)
            result.update(model=self.identity, query_key=query.key)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(result))
            temporary.replace(cache)
            return {**result, "cached": False}

    def close(self):
        with self._lock:
            if self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                self._process = None
            if self._stderr:
                self._stderr.close()
                self._stderr = None
