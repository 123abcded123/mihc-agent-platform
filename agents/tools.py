"""
工具注册表（对齐《项目文档》5.5：工具超时、参数校验、失败重试）

提供数据分析 Agent 使用的安全统计工具：
- 参数校验：数值列表长度/类型/范围；
- 超时控制：线程池 + 超时（Windows 兼容，不用 signal）；
- 失败重试：调用方负责。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import List, Dict, Any, Callable

from core.errors import AgentError

logger = logging.getLogger(__name__)


def _safe_describe(data: List[float]) -> Dict[str, float]:
    import statistics
    if not data:
        raise ValueError("数据为空")
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }


def _safe_ttest_ind(a: List[float], b: List[float]) -> Dict[str, float]:
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        raise ValueError("每组至少 2 个数据点")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p_value": float(p)}


def _safe_ttest_paired(a: List[float], b: List[float]) -> Dict[str, float]:
    from scipy import stats
    if len(a) < 2 or len(a) != len(b):
        raise ValueError("配对检验要求两组等长且至少 2 个数据点")
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p_value": float(p)}


def _safe_pearson(a: List[float], b: List[float]) -> Dict[str, float]:
    from scipy import stats
    if len(a) < 3 or len(a) != len(b):
        raise ValueError("相关分析要求两组等长且至少 3 个数据点")
    r, p = stats.pearsonr(a, b)
    return {"r": float(r), "p_value": float(p)}


class ToolRegistry:
    """工具注册与安全执行。"""

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec
        self._tools: Dict[str, Callable] = {
            "describe": _safe_describe,
            "ttest_ind": _safe_ttest_ind,
            "ttest_paired": _safe_ttest_paired,
            "pearson": _safe_pearson,
        }

    @property
    def available(self) -> List[str]:
        return list(self._tools.keys())

    @staticmethod
    def validate_numbers(data, max_len: int = 1000) -> List[float]:
        """参数校验：数值列表（长度/类型/范围）。"""
        if not isinstance(data, (list, tuple)):
            raise AgentError("参数校验失败: data 必须是数值列表")
        if len(data) > max_len:
            raise AgentError(f"参数校验失败: 数据点超过 {max_len} 上限")
        try:
            values = [float(x) for x in data]
        except (TypeError, ValueError):
            raise AgentError("参数校验失败: 列表包含非数值元素")
        if any(abs(v) > 1e15 for v in values):
            raise AgentError("参数校验失败: 数值超出合理范围")
        return values

    def run(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """执行工具（带参数校验与超时）。"""
        if tool_name not in self._tools:
            raise AgentError(f"工具不存在: {tool_name}，可用: {self.available}")

        if tool_name == "describe":
            data = self.validate_numbers(kwargs.get("data") or kwargs.get("a") or [])
            call = lambda: self._tools["describe"](data)  # noqa: E731
        else:
            a = self.validate_numbers(kwargs.get("a") or kwargs.get("data") or [])
            b = self.validate_numbers(kwargs.get("b") or [])
            call = lambda: self._tools[tool_name](a, b)  # noqa: E731

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(call)
                result = future.result(timeout=self.timeout_sec)
        except FutureTimeout:
            raise AgentError(f"工具 {tool_name} 执行超时（>{self.timeout_sec}s）")
        logger.info("Tool %s executed: %s", tool_name, result)
        return result
