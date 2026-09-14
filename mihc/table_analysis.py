"""mIHC/多组学表格的确定性校验与统计。

模型不参与列匹配、sample_id 对齐、阳性率或 p_value 计算。该模块返回可审计
的质量报告和聚合结果，报告 Agent 只能消费这些结果。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BASE_CELL_COLUMNS = {"sample_id", "cell_id", "group"}
BASE_OMICS_COLUMNS = {"sample_id", "feature", "value"}


def load_table(path: str, max_rows: int = 500000):
    import pandas as pd

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        # 客户文件可能来自 Excel 导出或 Windows 环境，按常见编码逐个尝试。
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"CSV 编码无法识别: {last_error}")
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的表格类型: {suffix}")
    if len(df) > max_rows:
        raise ValueError(f"行数超过上限 {max_rows}")
    df.columns = [str(column).strip() for column in df.columns]
    return df


def infer_table_kind(columns: Iterable[str]) -> str:
    columns = {str(column).strip() for column in columns}
    if BASE_CELL_COLUMNS.issubset(columns):
        return "cell_table"
    if BASE_OMICS_COLUMNS.issubset(columns):
        return "omics_table"
    return "unknown"


def _sample_map(project_samples: List[dict]) -> Dict[str, dict]:
    return {str(row["sample_id"]): row for row in project_samples}


def _resolve_column(columns: Iterable[str], name: str) -> Optional[str]:
    normalized = {str(column).strip().lower().replace("_", ""): column for column in columns}
    key = name.strip().lower().replace("_", "")
    return normalized.get(key)


def validate_cell_table(df, project_samples: List[dict], marker_specs: List[dict],
                        required_markers: Optional[List[str]] = None) -> dict:
    """返回结构化校验结果，不用异常替代用户可读的错误。"""
    errors: List[dict] = []
    warnings: List[dict] = []
    passed: List[str] = []
    columns = set(df.columns)
    missing = sorted(BASE_CELL_COLUMNS - columns)
    if missing:
        errors.append({"field": "columns", "reason": f"缺少字段: {missing}"})
    else:
        passed.append("columns")

    samples = _sample_map(project_samples)
    if not samples:
        errors.append({"field": "sample_id", "reason": "当前项目没有样本元数据，不能进行样本对齐"})
    elif "sample_id" in df:
        values = set(df["sample_id"].dropna().astype(str))
        unknown = sorted(values - set(samples))
        if unknown:
            errors.append({"field": "sample_id", "reason": f"存在不属于当前项目的样本: {unknown[:20]}"})
        else:
            passed.append("sample_id")

    if missing:
        return {"status": "failed", "passed_checks": passed, "failed_checks": ["columns"],
                "warnings": warnings, "errors": errors, "marker_columns": []}

    null_key_rows = int(df[["sample_id", "cell_id", "group"]].isna().any(axis=1).sum())
    if null_key_rows:
        errors.append({"field": "keys", "reason": f"sample_id/cell_id/group 存在 {null_key_rows} 行空值"})
    else:
        passed.append("required_values")

    duplicates = int(df.duplicated(subset=["sample_id", "cell_id"]).sum())
    if duplicates:
        errors.append({"field": "cell_id", "reason": f"存在 {duplicates} 个重复的 sample_id + cell_id"})
    else:
        passed.append("unique_cell_id")

    # group 必须和项目样本元数据一致，禁止按行顺序猜测样本对应关系。
    mismatches = []
    for sample_id, group in df[["sample_id", "group"]].dropna().astype(str).drop_duplicates().itertuples(index=False):
        expected = str(samples.get(sample_id, {}).get("group", ""))
        if expected and group != expected:
            mismatches.append({"sample_id": sample_id, "file_group": group, "project_group": expected})
    if mismatches:
        errors.append({"field": "group", "reason": "文件分组与项目样本元数据不一致", "details": mismatches[:20]})
    else:
        passed.append("group_definition")

    specs = {str(spec.get("name", "")).strip().lower(): spec for spec in marker_specs if spec.get("name")}
    requested = required_markers or list(specs)
    marker_columns: List[dict] = []
    for marker in requested:
        column = _resolve_column(columns, marker)
        if not column:
            errors.append({"field": marker, "reason": f"缺少 marker 列: {marker}"})
            continue
        if not __import__("pandas").api.types.is_numeric_dtype(df[column]):
            errors.append({"field": column, "reason": "marker 必须是数值列"})
            continue
        spec = specs.get(str(marker).lower(), {})
        threshold = spec.get("threshold")
        values = set(df[column].dropna().tolist())
        if threshold is None:
            if values and values.issubset({0, 1, 0.0, 1.0}):
                threshold = 0.5
                warnings.append({"field": column, "reason": "检测到二值列，使用 0.5 作为阳性阈值"})
            else:
                errors.append({"field": column, "reason": "未配置阈值，无法定义阳性率；请说明强度/概率阈值"})
                continue
        marker_columns.append({"name": marker, "column": column, "threshold": float(threshold)})
    if marker_columns:
        passed.append("marker_unit_and_threshold")

    return {
        "status": "failed" if errors else "passed",
        "passed_checks": passed,
        "failed_checks": sorted({error["field"] for error in errors}),
        "warnings": warnings,
        "errors": errors,
        "marker_columns": marker_columns,
        "row_count": int(len(df)),
    }


def _p_value(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy import stats
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return None


def summarize_cell_groups(df, marker_columns: List[dict]) -> dict:
    """先按 sample 聚合，再按 group 聚合，避免把细胞行当作独立生物学重复。"""
    import pandas as pd

    work = df.copy()
    for marker in marker_columns:
        work[marker["name"]] = pd.to_numeric(work[marker["column"]], errors="coerce")
        work[f"{marker['name']}_positive"] = work[marker["name"]] >= marker["threshold"]

    aggregations: Dict[str, Any] = {"cell_count": ("cell_id", "count")}
    for marker in marker_columns:
        aggregations[f"{marker['name']}_positive_rate"] = (f"{marker['name']}_positive", "mean")
    sample_level = work.groupby(["sample_id", "group"], dropna=False).agg(**aggregations).reset_index()

    results: List[dict] = []
    for marker in marker_columns:
        field = f"{marker['name']}_positive_rate"
        grouped = sample_level.groupby("group", dropna=False)[field]
        by_group = []
        values_by_group = {}
        for group, series in grouped:
            values = [float(value) for value in series.dropna().tolist()]
            values_by_group[str(group)] = values
            by_group.append({
                "group": str(group),
                "n_samples": len(values),
                "positive_rate_mean": float(sum(values) / len(values)) if values else None,
                "positive_rate_stdev": float(pd.Series(values).std(ddof=1)) if len(values) > 1 else 0.0,
            })
        groups = list(values_by_group)
        comparison = None
        if len(groups) == 2:
            comparison = {"groups": groups, "p_value": _p_value(values_by_group[groups[0]], values_by_group[groups[1]])}
        results.append({"marker": marker["name"], "metric": "positive_rate",
                        "threshold": marker["threshold"], "by_group": by_group,
                        "comparison": comparison})

    sample_count = {str(group): int(count) for group, count in sample_level.groupby("group")["sample_id"].nunique().items()}
    return {"sample_count": sample_count, "marker_results": results}


def summarize_omics(df, project_samples: List[dict], question: str = "") -> dict:
    import pandas as pd

    required = BASE_OMICS_COLUMNS - set(df.columns)
    if required:
        return {"status": "failed", "errors": [{"field": "columns", "reason": f"缺少字段: {sorted(required)}"}]}
    if not pd.api.types.is_numeric_dtype(df["value"]):
        return {"status": "failed", "errors": [{"field": "value", "reason": "value 必须是数值列"}]}
    samples = _sample_map(project_samples)
    unknown = sorted(set(df["sample_id"].dropna().astype(str)) - set(samples))
    if unknown:
        return {"status": "failed", "errors": [{"field": "sample_id", "reason": f"存在未知样本: {unknown[:20]}"}]}

    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["group"] = df["sample_id"].map(lambda value: samples[value].get("group", ""))
    requested = [marker for marker in ("IFNG", "IFN-γ") if marker.lower() in question.lower()]
    features = [str(value) for value in df["feature"].dropna().unique()]
    selected = [feature for feature in features if not requested or feature.lower() in {item.lower() for item in requested}]
    if not selected:
        selected = features[:20]

    output = []
    for feature in selected:
        feature_df = df[df["feature"].astype(str) == feature]
        sample_values = feature_df.groupby(["sample_id", "group"], dropna=False)["value"].mean().reset_index()
        by_group = []
        values_by_group = {}
        for group, series in sample_values.groupby("group")["value"]:
            values = [float(value) for value in series.dropna().tolist()]
            values_by_group[str(group)] = values
            by_group.append({"group": str(group), "n_samples": len(values),
                             "mean": float(sum(values) / len(values)) if values else None})
        groups = list(values_by_group)
        comparison = {"groups": groups, "p_value": _p_value(values_by_group[groups[0]], values_by_group[groups[1]])} if len(groups) == 2 else None
        output.append({"feature": feature, "by_group": by_group, "comparison": comparison})
    return {"status": "passed", "omics_results": output}


def analyze_tables(cell_df, omics_df, project_samples: List[dict], marker_specs: List[dict],
                   required_markers: Optional[List[str]], question: str) -> dict:
    cell_quality = validate_cell_table(cell_df, project_samples, marker_specs, required_markers)
    omics_quality = summarize_omics(omics_df, project_samples, question) if omics_df is not None else None
    if cell_quality["status"] != "passed":
        return {"status": "failed", "quality_report": {"cell_table": cell_quality, "omics_table": omics_quality},
                "errors": cell_quality["errors"], "limitations": ["细胞表格未通过质量校验"]}
    if omics_quality and omics_quality.get("status") != "passed":
        return {"status": "failed", "quality_report": {"cell_table": cell_quality, "omics_table": omics_quality},
                "errors": omics_quality.get("errors", []), "limitations": ["组学表格未通过质量校验"]}

    summary = summarize_cell_groups(cell_df, cell_quality["marker_columns"])
    return {
        "status": "passed",
        "quality_report": {"cell_table": cell_quality, "omics_table": omics_quality},
        "sample_count": summary["sample_count"],
        "marker_results": summary["marker_results"],
        "omics_results": (omics_quality or {}).get("omics_results", []),
        "limitations": ["统计在样本级聚合后进行，不能替代完整实验设计审查",
                        "结果表示观察和统计关联，不自动证明因果关系"],
    }
