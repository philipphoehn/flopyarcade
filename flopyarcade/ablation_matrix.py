"""B20: Ablation matrix for reward component impact analysis.

Remove one reward component at a time across environments and measure
the resulting score delta to rank components by impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardComponent:
    """A single named reward component with optional weight and description."""

    name: str
    weight: float = 1.0
    description: str = ''


def compute_ablation_scores(
    base_scores: dict[str, float],
    component_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute per-envtype delta = base - removed.

    Parameters
    ----------
    base_scores:
        Mapping of envtype to full (all-components) score.
    component_scores:
        Mapping of envtype to {component_name: score_with_component_removed}.

    Returns
    -------
    dict[str, float]
        Mapping of envtype to mean delta across all components for that env.
        (If a single component is queried the delta is that component's delta.)
        Actually: returns {envtype: delta} where delta = base_score - average of
        the component_scores values for that envtype.
    """
    result: dict[str, float] = {}
    for envtype, base in base_scores.items():
        comp_scores = component_scores.get(envtype, {})
        if not comp_scores:
            result[envtype] = 0.0
            continue
        # delta = base - average of all component-removed scores
        avg_removed = sum(comp_scores.values()) / len(comp_scores)
        result[envtype] = base - avg_removed
    return result


def build_ablation_table(
    envtypes: list[str],
    components: list[RewardComponent],
    scores_fn,
) -> list[dict]:
    """Build the full ablation table.

    Parameters
    ----------
    envtypes:
        List of environment type identifiers.
    components:
        List of RewardComponent objects to ablate.
    scores_fn:
        Callable (envtype, components_to_keep: list[str]) -> float.
        Called with the full set minus one component to get the ablated score.
        Called with the full set to get the full score.

    Returns
    -------
    list[dict]
        Rows sorted by |delta| descending. Each row has keys:
        envtype, component, score_full, score_ablated, delta, pct_change.
    """
    all_names = [c.name for c in components]
    rows: list[dict] = []

    for envtype in envtypes:
        full_score = scores_fn(envtype, list(all_names))
        for comp in components:
            remaining = [n for n in all_names if n != comp.name]
            ablated_score = scores_fn(envtype, remaining)
            delta = full_score - ablated_score
            if full_score != 0:
                pct_change = (delta / abs(full_score)) * 100.0
            else:
                pct_change = 0.0
            rows.append({
                'envtype': envtype,
                'component': comp.name,
                'score_full': full_score,
                'score_ablated': ablated_score,
                'delta': delta,
                'pct_change': pct_change,
            })

    rows.sort(key=lambda r: abs(r['delta']), reverse=True)
    return rows


def rank_components(table: list[dict]) -> list[str]:
    """Return component names ranked by impact (largest |delta| first), deduplicated.

    A component appears once, at the position of its first (largest |delta|) occurrence.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for row in table:
        name = row['component']
        if name not in seen:
            seen.add(name)
            ranked.append(name)
    return ranked


def find_critical_components(
    table: list[dict],
    threshold_pct: float = 10.0,
) -> list[str]:
    """Return components whose removal causes > threshold_pct% score drop.

    A component is critical if ANY row for it has pct_change > threshold_pct.
    """
    critical: set[str] = set()
    for row in table:
        if row['pct_change'] > threshold_pct:
            critical.add(row['component'])
    # preserve table order (already sorted by |delta| desc)
    seen: set[str] = set()
    result: list[str] = []
    for row in table:
        name = row['component']
        if name in critical and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def format_ablation_report(table: list[dict]) -> str:
    """Return a human-readable text report of the ablation table."""
    if not table:
        return "Ablation Report\n===============\nNo data available.\n"

    lines: list[str] = []
    lines.append("Ablation Report")
    lines.append("===============")
    lines.append("")

    # Group by envtype for readability
    envtypes: list[str] = []
    for row in table:
        if row['envtype'] not in envtypes:
            envtypes.append(row['envtype'])

    for env in envtypes:
        env_rows = [r for r in table if r['envtype'] == env]
        lines.append(f"Environment: {env}")
        lines.append(f"  {'Component':<20} {'Full':>10} {'Ablated':>10} {'Delta':>10} {'PctChg':>8}")
        lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for r in env_rows:
            lines.append(
                f"  {r['component']:<20} {r['score_full']:>10.4f} {r['score_ablated']:>10.4f} {r['delta']:>10.4f} {r['pct_change']:>7.2f}%"
            )
        lines.append("")

    # Ranking summary
    ranked = rank_components(table)
    lines.append("Component Impact Ranking (by |delta|):")
    for i, name in enumerate(ranked, 1):
        lines.append(f"  {i}. {name}")

    critical = find_critical_components(table)
    if critical:
        lines.append("")
        lines.append(f"Critical components (>10% drop): {', '.join(critical)}")
    else:
        lines.append("")
        lines.append("No critical components identified.")

    return '\n'.join(lines)
