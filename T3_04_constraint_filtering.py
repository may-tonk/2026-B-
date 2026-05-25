import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


TOLERANCE = 1e-4
MARGIN_LOWER = 0.0
MARGIN_UPPER = 8.0


def load_inputs():
    eval_path = Path('model/T3_03_price_evaluation_results.pkl')
    cfg_path = Path('model/T3_01_financial_model_config.pkl')
    if not eval_path.exists() or not cfg_path.exists():
        print('[ERROR] Missing T3_03 evaluation results or T3_01 financial config.')
        print('Run T3_01, T3_02 and T3_03 first.')
        sys.exit(1)

    with eval_path.open('rb') as eval_file:
        evaluated_results = pickle.load(eval_file)
    with cfg_path.open('rb') as config_file:
        financial_config = pickle.load(config_file)
    return evaluated_results, financial_config


def check_constraints(evaluated_plan):
    """Check hard constraints under the station-level break-even margin rule."""
    capacity_ok = bool(evaluated_plan['capacity_feasible'])
    system_margin = evaluated_plan['system_financials']['profit_margin']
    system_margin_ok = MARGIN_LOWER <= system_margin <= MARGIN_UPPER

    subsidy_cap_stations = []
    station_margin_failed = []
    station_margin_ok = True

    for station_idx, station_financials in evaluated_plan['station_financials'].items():
        if (
            station_financials['daily_subsidy_request_yuan']
            > station_financials['daily_subsidy_cap_yuan'] + TOLERANCE
        ):
            subsidy_cap_stations.append(station_idx)

        station_margin = station_financials['profit_margin']
        if station_margin < MARGIN_LOWER - TOLERANCE or station_margin > MARGIN_UPPER + TOLERANCE:
            station_margin_failed.append(station_idx)
            station_margin_ok = False

    return {
        'capacity_ok': capacity_ok,
        'station_margin_ok': station_margin_ok,
        'system_margin_ok': system_margin_ok,
        'hard_feasible': capacity_ok and station_margin_ok,
        'subsidy_cap_stations': subsidy_cap_stations,
        'station_margin_failed': station_margin_failed,
    }


def station_label(station_idx, financial_config):
    scale = financial_config['station_scale_names_zh'][station_idx]
    return f'station {station_idx} ({scale})'


def print_counter(counter, financial_config, title):
    print(title)
    if not counter:
        print('  none\n')
        return
    for station_idx, count in counter.most_common():
        print(f'  - {station_label(station_idx, financial_config)}: {count:,}')
    print()


def describe_price_tuple(plan, paid_services, active_stations):
    values = list(plan['price_plan_tuple'])
    service_count = len(paid_services)
    chunks = []
    for station_pos, station_idx in enumerate(active_stations):
        start = station_pos * service_count
        stop = start + service_count
        chunks.append(f'{station_idx}:{values[start:stop]}')
    return ' | '.join(chunks)


def main():
    print('========== T3_04: station-level constraint filtering ==========\n')
    evaluated_results, financial_config = load_inputs()
    total_plans = len(evaluated_results)
    active_stations = list(financial_config['active_stations'])
    paid_services = list(financial_config['paid_services'])

    print('[hard constraints]')
    print(f'  1. each active station profit margin in [{MARGIN_LOWER:.0f}%, {MARGIN_UPPER:.0f}%]')
    print(
        f"  2. station load <= attachment capacity x "
        f"{financial_config['capacity_limit_ratio']:.2f} inherited from T2 elastic capacity"
    )
    print('\n[audit only]')
    print('  1. system-wide profit margin is reported but not used as an extra hard constraint')
    print('  2. subsidy requests above daily caps are truncated by min(request, cap)\n')

    system_margin_below_zero = 0
    system_margin_above_eight = 0
    system_margin_ok_count = 0
    station_margin_ok_count = 0
    capacity_ok_count = 0
    capacity_failed_count = 0
    subsidy_cap_counter = Counter()
    station_margin_failed_counter = Counter()
    feasible_plans = []

    for plan in evaluated_results:
        system_margin = plan['system_financials']['profit_margin']
        if system_margin < MARGIN_LOWER:
            system_margin_below_zero += 1
        elif system_margin > MARGIN_UPPER:
            system_margin_above_eight += 1
        else:
            system_margin_ok_count += 1

        audit = check_constraints(plan)
        capacity_ok_count += int(audit['capacity_ok'])
        capacity_failed_count += int(not audit['capacity_ok'])
        station_margin_ok_count += int(audit['station_margin_ok'])
        subsidy_cap_counter.update(audit['subsidy_cap_stations'])
        station_margin_failed_counter.update(audit['station_margin_failed'])

        plan['constraint_audit'] = audit
        plan['is_feasible'] = audit['hard_feasible']
        if audit['hard_feasible']:
            feasible_plans.append(plan)

    print('[filtering summary]')
    print(f'  candidate plans: {total_plans:,}')
    print(f'  station-margin feasible plans: {station_margin_ok_count:,} ({station_margin_ok_count / total_plans * 100:.2f}%)')
    print(f'  elastic-capacity feasible plans: {capacity_ok_count:,} ({capacity_ok_count / total_plans * 100:.2f}%)')
    print(f'  elastic-capacity failed plans: {capacity_failed_count:,} ({capacity_failed_count / total_plans * 100:.2f}%)')
    print(f'  hard-feasible plans: {len(feasible_plans):,} ({len(feasible_plans) / total_plans * 100:.2f}%)')
    print('\n[system-wide margin audit]')
    print(f'  system margin < 0%: {system_margin_below_zero:,}')
    print(f'  system margin in [0%, 8%]: {system_margin_ok_count:,}')
    print(f'  system margin > 8%: {system_margin_above_eight:,}\n')

    print_counter(subsidy_cap_counter, financial_config, '[subsidy cap reached]')
    print_counter(station_margin_failed_counter, financial_config, '[station margin failed]')

    if feasible_plans:
        feasible_satisfactions = [plan['weighted_avg_satisfaction'] for plan in feasible_plans]
        feasible_system_margins = [plan['system_financials']['profit_margin'] for plan in feasible_plans]
        highest_satisfaction_plan = feasible_plans[int(np.argmax(feasible_satisfactions))]
        station_margins = {
            station_idx: highest_satisfaction_plan['station_financials'][station_idx]['profit_margin']
            for station_idx in active_stations
        }
        print('[feasible region]')
        print(f'  satisfaction range: {min(feasible_satisfactions):.4f} ~ {max(feasible_satisfactions):.4f}')
        print(f'  system margin range: {min(feasible_system_margins):.2f}% ~ {max(feasible_system_margins):.2f}%')
        print(f'  highest-satisfaction plan id: {highest_satisfaction_plan["plan_id"]}')
        print(f'  station margins: {station_margins}')
        print(f'  prices: {describe_price_tuple(highest_satisfaction_plan, paid_services, active_stations)}\n')
    else:
        print('[feasible region]')
        print('  no plan satisfies station-level margin and elastic capacity constraints.\n')

    os.makedirs('model', exist_ok=True)
    output_path = Path('model/T3_04_feasible_price_plans.pkl')
    with output_path.open('wb') as output_file:
        pickle.dump(feasible_plans, output_file)

    print(f'Saved feasible plans to: {output_path}')
    print('============================================================\n')


if __name__ == '__main__':
    main()
