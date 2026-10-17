"""One-click benchmark runner: policy vs. the full heuristic zoo.

:func:`run_benchmark` takes a resolved GRACO config, builds the env + generator,
freezes a fixed benchmark set of graphs, then runs **every applicable heuristic**
in :data:`~graco.heuristics.base.HEURISTICS` over it (optionally filtered by
name).  If a checkpoint is supplied, the learned policy is evaluated with a
greedy rollout on the *same* graphs, so its number lands in the same table.

Results are ranked by the env's primary metric: :attr:`VectorizedEnv.eval_metric`
picks the ranking column (``"objective"`` vs. ``"return"`` — dismantling ranks by
return = -ANC) and :attr:`VectorizedEnv.maximize` picks the sort direction.

    from graco.eval.benchmark import run_benchmark
    from graco.utils.config import load_config
    run_benchmark(load_config("finder"), ckpt="runs/finder/best.pt")
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from graco.eval.decoding import decode
from graco.heuristics.base import HEURISTICS
from graco.registries import ALGOS, ENVS, GENERATORS
from graco.utils.config import to_container
from graco.utils.seeding import resolve_device, seed_everything

# columns reported per method
_FIELDS = ("objective_mean", "objective_std", "return_mean", "seconds")


def _register_families() -> None:
    """Import every pluggable family so its ``@register`` decorators run.

    Each import is isolated: a missing optional module (e.g. a heuristic file
    that has not been added yet) never aborts the whole benchmark.
    """
    modules = [
        "graco.envs",
        "graco.generators",
        "graco.models",
        "graco.algos",
        "graco.buffers",
        "graco.baselines",
        "graco.heuristics.greedy",
        "graco.heuristics.annealing",
        "graco.heuristics.local_search",
        "graco.heuristics.community",
        "graco.heuristics.solvers",
    ]
    for mod in modules:
        try:
            __import__(mod)
        except Exception as exc:  # pragma: no cover - defensive, optional deps
            print(f"[benchmark] skipping '{mod}': {exc}")


def _unique_heuristics() -> Dict[type, List[str]]:
    """Map each registered heuristic class to the registry keys pointing at it."""
    class_keys: Dict[type, List[str]] = {}
    for key in HEURISTICS.keys():
        cls = HEURISTICS.get(key)
        class_keys.setdefault(cls, []).append(key)
    return class_keys


def run_benchmark(
    cfg: DictConfig,
    ckpt: Optional[str] = None,
    num: Optional[int] = None,
    methods: Optional[List[str]] = None,
    device: Optional[str] = None,
    decode_strategy: str = "greedy",
    samples: int = 8,
    temperature: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Benchmark all applicable heuristics (and an optional policy) on one config.

    Args:
        cfg: resolved config (``env`` / ``generator`` / ``algo`` / ``trainer``).
        ckpt: path to a checkpoint (``torch.save`` dict with an ``"algo"`` key);
            if given, its policy is evaluated with a greedy rollout.
        num: number of benchmark graphs (default ``cfg.trainer.n_valid`` or 128).
        methods: restrict heuristics to these names (registry key or ``.name``).
        device: override device (default ``cfg.device`` -> auto).

    Returns:
        ``{method: {objective_mean, objective_std, return_mean, seconds}}``.
        Also prints the ranked table to stdout as a side effect.
    """
    _register_families()

    dev = resolve_device(device if device is not None else cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 0))
    seed_everything(seed, deterministic=bool(cfg.get("deterministic", False)))

    env = ENVS.build(to_container(cfg.env))
    generator = GENERATORS.build(to_container(cfg.generator))

    tcfg = cfg.get("trainer", {}) or {}
    default_num = int(tcfg.get("n_valid", 128)) if tcfg else 128
    num = int(num) if num is not None else default_num

    # fixed benchmark set (same seed offset as the trainer's val set for parity)
    rng = np.random.default_rng(seed + 12345)
    graphs = generator.sample(num, device=dev, rng=rng)

    results: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------- heuristics
    for cls, keys in _unique_heuristics().items():
        try:
            heuristic = cls()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[benchmark] cannot instantiate {cls.__name__}: {exc}")
            continue
        label = getattr(heuristic, "name", None) or keys[0]
        if label == "heuristic":  # base default -> prefer the registry key
            label = keys[0]
        aliases = set(keys) | {label}
        if methods is not None and not (aliases & set(methods)):
            continue
        if not heuristic.applicable(env, graphs):
            continue
        try:
            res = heuristic.solve(env, graphs.clone())
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[benchmark] {label} failed: {exc}")
            continue
        obj = res.objective.float()
        results[label] = {
            "objective_mean": float(obj.mean()),
            "objective_std": float(obj.std()),
            "return_mean": float(res.ret.float().mean()),
            "seconds": float(res.seconds),
        }

    # ----------------------------------------------------------------- policy
    if ckpt is not None:
        algo_cfg = to_container(cfg.algo)
        algo = ALGOS.build(algo_cfg, env=env, device=str(dev))
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        algo.load_state_dict(state["algo"])
        label = f"RL({algo_cfg.get('type', 'policy')}"
        label += f"/{decode_strategy}x{samples})" if decode_strategy != "greedy" else ")"
        t0 = time.time()
        obj, ret = decode(
            env, graphs.clone(), algo, strategy=decode_strategy,
            samples=samples, temperature=temperature,
        )
        seconds = time.time() - t0
        obj = obj.float()
        results[label] = {
            "objective_mean": float(obj.mean()),
            "objective_std": float(obj.std()),
            "return_mean": float(ret.float().mean()),
            "seconds": float(seconds),
        }

    print(format_table(results, env))
    return results


def format_table(results: Dict[str, Dict[str, float]], env) -> str:
    """Render ``results`` as an aligned text table ranked by the env metric."""
    metric = getattr(env, "eval_metric", "objective")
    maximize = bool(getattr(env, "maximize", True))
    rank_key = "return_mean" if metric == "return" else "objective_mean"

    title = (
        f"Benchmark: {type(env).__name__}  |  rank by {metric} "
        f"({'higher' if maximize else 'lower'} is better)"
    )
    if not results:
        return title + "\n(no applicable methods)"

    worst = float("-inf") if maximize else float("inf")
    order = sorted(
        results.items(),
        key=lambda kv: kv[1].get(rank_key, worst),
        reverse=maximize,
    )

    best = order[0][1].get(rank_key, 0.0)

    def gap_pct(v: float) -> float:
        if best == 0:
            return 0.0 if v == best else float("inf")
        return ((best - v) if maximize else (v - best)) / abs(best) * 100.0

    headers = ["", "method", "objective", "obj.std", "return", "gap%", "time(s)"]
    rank_col = 2 if rank_key == "objective_mean" else 4
    headers[rank_col] += " *"

    rows = []
    for i, (name, m) in enumerate(order):
        marker = "->" if i == 0 else ""
        rows.append(
            [
                marker,
                name,
                f"{m['objective_mean']:.4f}",
                f"{m['objective_std']:.4f}",
                f"{m['return_mean']:.4f}",
                f"{gap_pct(m.get(rank_key, 0.0)):.2f}",
                f"{m['seconds']:.3f}",
            ]
        )

    widths = [
        max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(len(headers))
    ]

    def _fmt(cells: List[str]) -> str:
        out = []
        for c, cell in enumerate(cells):
            out.append(cell.ljust(widths[c]) if c <= 1 else cell.rjust(widths[c]))
        return "  ".join(out).rstrip()

    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    lines = [title, sep, _fmt(headers), sep]
    lines += [_fmt(r) for r in rows]
    lines.append(sep)
    lines.append("gap% = shortfall vs the best method above (0 = best; exact solver present => optimality gap)")
    return "\n".join(lines)
