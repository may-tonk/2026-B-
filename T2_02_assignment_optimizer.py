from dataclasses import dataclass


OBJECTIVE_TOLERANCE = 1e-10
CAPACITY_TOLERANCE = 1e-6
SATISFACTION_TOLERANCE = 1e-10


@dataclass(frozen=True)
class AssignmentOption:
    """One reachable single-station choice for a community."""

    community_idx: int
    station_idx: int
    satisfaction: float
    effective_load: float
    distance: float


def _assignment_tuple(assignments, num_communities):
    return tuple(assignments.get(community_idx, -1) for community_idx in range(num_communities))


def _is_better_solution(candidate, best, num_communities):
    """Compare complete search results deterministically."""
    if best is None:
        return True

    objective_diff = candidate['objective'] - best['objective']
    if objective_diff > OBJECTIVE_TOLERANCE:
        return True
    if abs(objective_diff) > OBJECTIVE_TOLERANCE:
        return False

    candidate_max_utilization = max(candidate['station_utilizations'].values(), default=0.0)
    best_max_utilization = max(best['station_utilizations'].values(), default=0.0)
    if candidate_max_utilization < best_max_utilization - CAPACITY_TOLERANCE:
        return True
    if candidate_max_utilization > best_max_utilization + CAPACITY_TOLERANCE:
        return False

    return _assignment_tuple(candidate['assignments'], num_communities) < _assignment_tuple(
        best['assignments'],
        num_communities,
    )


def _package_solution(
    assignments,
    chosen_options,
    station_loads,
    station_capacities,
    capacity_limit_ratio,
    objective,
    num_communities,
):
    community_satisfactions = [0.0] * num_communities
    community_effective_daily_visits = [0.0] * num_communities
    for community_idx, option in chosen_options.items():
        community_satisfactions[community_idx] = option.satisfaction
        community_effective_daily_visits[community_idx] = option.effective_load

    station_utilizations = {
        station_idx: station_loads[station_idx] / station_capacities[station_idx]
        for station_idx in station_loads
    }
    return {
        'assignments': dict(assignments),
        'community_satisfactions': community_satisfactions,
        'community_effective_daily_visits': community_effective_daily_visits,
        'station_loads': dict(station_loads),
        'station_utilizations': station_utilizations,
        'capacity_limit_ratio': float(capacity_limit_ratio),
        'objective': float(objective),
    }


def optimize_capacity_constrained_assignments(
    options_by_community,
    station_capacities,
    satisfaction_weights,
    num_communities,
    capacity_limit_ratio=1.0,
):
    """Search capacity-feasible assignments inside tied highest-score choices.

    The outer Scheme B solver must filter out non-maximum satisfaction options
    before this helper runs. This helper may choose between tied highest-score
    stations under attachment capacities, but it must never send a community to
    a lower-satisfaction station.
    """
    assigned_communities = [
        community_idx
        for community_idx, options in options_by_community.items()
        if options
    ]
    if not assigned_communities:
        return _package_solution(
            {},
            {},
            {station_idx: 0.0 for station_idx in station_capacities},
            station_capacities,
            capacity_limit_ratio,
            0.0,
            num_communities,
        )

    ordered_options = {}
    max_remaining_objective = {}
    for community_idx in assigned_communities:
        community_options = options_by_community[community_idx]
        option_satisfactions = [option.satisfaction for option in community_options]
        if max(option_satisfactions) - min(option_satisfactions) > SATISFACTION_TOLERANCE:
            raise ValueError(
                f'小区 {community_idx} 的归属候选包含非最高满意度站点'
            )

        options = sorted(
            community_options,
            key=lambda option: (
                -option.satisfaction,
                option.effective_load,
                option.distance,
                option.station_idx,
            ),
        )
        ordered_options[community_idx] = options
        max_remaining_objective[community_idx] = (
            satisfaction_weights[community_idx] * options[0].satisfaction
        )

    search_order = sorted(
        assigned_communities,
        key=lambda community_idx: (
            len(ordered_options[community_idx]),
            -min(option.effective_load for option in ordered_options[community_idx]),
            community_idx,
        ),
    )

    suffix_upper_bounds = [0.0] * (len(search_order) + 1)
    for search_pos in range(len(search_order) - 1, -1, -1):
        community_idx = search_order[search_pos]
        suffix_upper_bounds[search_pos] = (
            suffix_upper_bounds[search_pos + 1]
            + max_remaining_objective[community_idx]
        )

    station_loads = {station_idx: 0.0 for station_idx in station_capacities}
    assignments = {}
    chosen_options = {}
    best = None

    def search(search_pos, objective):
        nonlocal best

        if best is not None and objective + suffix_upper_bounds[search_pos] < best['objective'] - OBJECTIVE_TOLERANCE:
            return

        if search_pos == len(search_order):
            candidate = _package_solution(
                assignments,
                chosen_options,
                station_loads,
                station_capacities,
                capacity_limit_ratio,
                objective,
                num_communities,
            )
            if _is_better_solution(candidate, best, num_communities):
                best = candidate
            return

        community_idx = search_order[search_pos]
        objective_weight = satisfaction_weights[community_idx]
        for option in ordered_options[community_idx]:
            current_load = station_loads[option.station_idx]
            next_load = current_load + option.effective_load
            capacity_limit = station_capacities[option.station_idx] * capacity_limit_ratio
            if next_load > capacity_limit + CAPACITY_TOLERANCE:
                continue

            station_loads[option.station_idx] = next_load
            assignments[community_idx] = option.station_idx
            chosen_options[community_idx] = option

            search(
                search_pos + 1,
                objective + objective_weight * option.satisfaction,
            )

            station_loads[option.station_idx] = current_load
            assignments.pop(community_idx)
            chosen_options.pop(community_idx)

    search(0, 0.0)
    return best
