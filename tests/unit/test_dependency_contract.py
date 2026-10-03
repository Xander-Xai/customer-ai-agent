"""依赖声明契约（可复现性防回归）。

背景：本仓库曾在 **没有声明** ``prometheus_client`` 的情况下依赖它。它在
``core/monitoring.py`` / ``cache/response_cache.py`` 里是 try/except 包裹的**运行时**
import，缺失时应用照常启动，但 ``/api/metrics`` 变空、所有 ``agent_*`` 指标静默消失。
这种"静默降级"只有靠**显式声明依赖**才能避免。

这些测试锁住：
  1. 运行时依赖集合（requirements.txt）必须覆盖代码里 import 的第三方顶层模块；
  2. 允许的例外必须**显式**写进 optional/legacy 白名单；
  3. 运行时依赖不得包含任何真实凭据。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REQ = ROOT / "requirements.txt"
REQ_DEV = ROOT / "requirements-dev.txt"
REQ_OPT = ROOT / "requirements-optional.txt"

#: 允许"被 import 但不直接声明"的模块及其理由。空集之外的新增必须显式登记。
ALLOWED_UNDECLARED: dict[str, str] = {
    "opentelemetry": (
        "requirements.txt 中刻意注释掉（OPENTELEMETRY_ENABLED 默认关闭）；"
        "core/tracing.py 有优雅降级"
    ),
}

#: 由已声明发行包**传递**提供、不需要单独声明的 import 名 -> 提供方（任一即可）。
#: 显式列出是为了让"这个 import 为什么不用声明"可被 review，而不是靠运气。
TRANSITIVE_PROVIDERS: dict[str, str | tuple[str, ...]] = {
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "docx": "python-docx",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "cryptography": "python-jose",
    "numpy": ("qdrant-client", "openai", "opencv-python-headless"),
    "requests": ("qdrant-client", "langchain-community", "edge-tts"),
    "multipart": "python-multipart",
    "jose": "python-jose",
    # tools/mcp_adapter.py 用 anyio.fail_after 给 MCP 握手加超时。anyio 是
    # starlette（FastAPI 依赖）与官方 mcp SDK 共同传递提供的硬依赖，二者都已在
    # requirements*.txt 中声明，故不必单独声明 anyio。
    "anyio": ("starlette", "mcp"),
}

#: 这些模块由仓库自身提供，不是第三方依赖。
_LOCAL_ROOTS = {
    p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")
} | {p.stem for p in ROOT.glob("*.py")}


def _declared_names(*files: Path) -> set[str]:
    names: set[str] = set()
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
                if sep in line:
                    names.add(line.split(sep)[0].strip().lower().replace("_", "-"))
                    break
    return names


def _runtime_third_party_imports() -> set[str]:
    """源码（不含 tests/）里 import 的第三方顶层模块。"""
    stdlib = set(sys.stdlib_module_names)
    found: set[str] = set()
    for f in ROOT.rglob("*.py"):
        rel = f.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", ".venv/", "node_modules/", ".worktrees/", ".claude/")):
            continue
        if rel.startswith("scripts/"):
            # 脚本允许使用 requirements-optional.txt 的延迟导入依赖
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                found.add(node.module.split(".")[0])
    return {m for m in found if m and m not in stdlib and m not in _LOCAL_ROOTS}


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _is_declared(mod: str, declared: set[str]) -> bool:
    canonical = _canonical(mod)
    if any(canonical == d or canonical == d.split("-")[0] for d in declared):
        return True
    provider = TRANSITIVE_PROVIDERS.get(mod)
    if not provider:
        return False
    providers = (provider,) if isinstance(provider, str) else provider
    return any(_canonical(p) in declared for p in providers)


@pytest.mark.unit
def test_runtime_imports_are_declared():
    """运行时 import 的第三方模块必须在 requirements*.txt 中声明。"""
    declared = _declared_names(REQ, REQ_DEV, REQ_OPT)
    undeclared = []
    for mod in sorted(_runtime_third_party_imports()):
        if _is_declared(mod, declared):
            continue
        if mod in ALLOWED_UNDECLARED:
            continue
        undeclared.append(mod)
    assert not undeclared, (
        "运行时 import 但未在 requirements*.txt 声明的模块："
        f"{undeclared}。要么声明依赖，要么（若确为可选/历史）登记到 "
        "ALLOWED_UNDECLARED 并说明理由 —— 否则会像 prometheus_client 那样静默降级。"
    )


@pytest.mark.unit
def test_prometheus_client_is_declared():
    """回归锁定：prometheus_client 曾缺失导致所有指标静默变 no-op。"""
    declared = _declared_names(REQ, REQ_DEV, REQ_OPT)
    assert "prometheus-client" in declared, (
        "requirements.txt 必须显式声明 prometheus-client；"
        "它在 core/monitoring.py / cache/response_cache.py 是 try/except 的运行时 import，"
        "缺失时应用能启动但 /api/metrics 为空、所有 agent_* 指标消失"
    )


@pytest.mark.unit
def test_allowed_undeclared_entries_are_documented():
    """白名单里的每个模块都必须真的"没被声明"，否则该从白名单移除。"""
    declared = _declared_names(REQ, REQ_DEV, REQ_OPT)
    for mod, reason in ALLOWED_UNDECLARED.items():
        assert reason.strip(), f"{mod} 的豁免必须写明理由"
        assert not _is_declared(mod, declared), (
            f"{mod} 现在已被正式声明或传递提供，应从 ALLOWED_UNDECLARED 移除"
        )


@pytest.mark.unit
def test_runtime_requirements_contain_no_credentials():
    """requirements 不得含真实凭据。"""
    secretish = re.compile(
        r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}"
    )
    for f in (REQ, REQ_DEV, REQ_OPT):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert not secretish.search(line), f"{f.name}:{i} 疑似真实凭据: {line.strip()}"


@pytest.mark.unit
def test_dev_requirements_pin_the_asyncio_floor():
    """保持既有契约：pytest-asyncio 下限 >= 0.23.4（pytest>=8 兼容）。"""
    text = REQ_DEV.read_text(encoding="utf-8")
    m = re.search(r"pytest-asyncio\s*>=\s*([0-9.]+)", text)
    assert m, "requirements-dev.txt 必须声明 pytest-asyncio"
    parts = tuple(int(x) for x in m.group(1).split("."))
    assert parts >= (0, 23, 4), f"pytest-asyncio 下限过低: {m.group(1)}"


@pytest.mark.unit
def test_repo_does_not_ship_a_venv():
    """仓库不得提交 .venv（应由各环境自行创建）。"""
    tracked = subprocess_git_ls_files()
    offenders = [p for p in tracked if p.startswith(".venv/") or "/.venv/" in p]
    assert not offenders, f"仓库内不应跟踪虚拟环境: {offenders[:5]}"


def subprocess_git_ls_files() -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    return proc.stdout.splitlines()
