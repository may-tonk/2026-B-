import itertools
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np


SEARCH_LOWER = 0.60
SEARCH_UPPER = 1.00
COARSE_STEP = 0.02
FINE_STEP = 0.001
LOCAL_PADDING = 0.015
MARGIN_LOWER = 0.0
MARGIN_UPPER = 8.0
DAYS_PER_YEAR = 365


def load_pickle(path):
    with Path(path).open('rb') as file:
        return pickle.load(file)


def load_inputs():
    cfg_path = Path('model/T3_01_financial_model_config.pkl')
    t2_path = Path('model/T2_04_best_solution.pkl')
    if not cfg_path.exists() or not t2_path.exists():
        print('[ERROR] Missing T3_01 config or T2_04 best solution. Run T3_01 first.')
        sys.exit(1)
    return load_pickle(cfg_path), load_pickle(t2_path)


def rounded_range(start, stop, step):
    values = []
    current = start
    while current <= stop + step / 2:
        values.append(round(current, 3))
        current += step
    return values


def build_price_vector(base_prices, multiplier):
    return tuple(round(float(price) * float(multiplier), 2) for price in base_prices)


def package_station_template(template_id, services, base_prices, multiplier, family):
    prices = {
        service: price
        for service, price in zip(services, build_price_vector(base_prices, multiplier))
    }
    prices['紧急救助'] = 0.0
    return {
        'template_id': template_id,
        'family': family,
        'label': f'all paid services {multiplier:.3f}x base price',
        'multipliers': tuple([float(multiplier)] * len(services)),
        'uniform_multiplier': float(multiplier),
        'prices': prices,
        'price_tuple': tuple(prices[service] for service in services),
    }


def station_base_revenue_wan(station_daily_by_service, base_prices):
    paid_daily = np.asarray(station_daily_by_service[:len(base_prices)], dtype=float)
    base_prices = np.asarray(base_prices, dtype=float)
    return float(np.sum(paid_daily * base_prices) * DAYS_PER_YEAR / 10000.0)


def station_variable_cost_wan(station_daily_by_service, direct_costs):
    daily_by_service = np.asarray(station_daily_by_service, dtype=float)
    direct_costs = np.asarray(direct_costs, dtype=float)
    return float(np.sum(daily_by_service * direct_costs) * DAYS_PER_YEAR / 10000.0)


def station_subsidy_wan(station_daily_by_service, subsidy_per_visit, daily_cap_yuan, paid_count):
    paid_daily = float(np.sum(np.asarray(station_daily_by_service[:paid_count], dtype=float)))
    daily_subsidy = min(paid_daily * subsidy_per_visit, float(daily_cap_yuan))
    return daily_subsidy * DAYS_PER_YEAR / 10000.0


def estimate_station_margin_interval(station_idx, financial_config, t2_best, base_prices):
    """Estimate the station-uniform multiplier interval that yields 0%-8% margin.

    For prices not above the base price, S3 stays at 1.0, so the T2 effective
    service volume is a reliable local estimate. T3_03 still re-evaluates every
    generated plan exactly before filtering.
    """
    daily_by_service = t2_best['station_effective_daily_by_service'][station_idx]
    annual_base_revenue = station_base_revenue_wan(daily_by_service, base_prices)
    annual_variable_cost = station_variable_cost_wan(
        daily_by_service,
        financial_config['direct_costs'],
    )
    annual_subsidy = station_subsidy_wan(
        daily_by_service,
        financial_config['subsidy_per_visit'],
        financial_config['daily_subsidy_caps_yuan'][station_idx],
        len(base_prices),
    )
    annual_operating_cost = (
        financial_config['daily_fixed_costs_yuan'][station_idx] * DAYS_PER_YEAR / 10000.0
        + financial_config['station_annual_depreciation_wan'][station_idx]
    )

    if annual_base_revenue <= 0:
        return None

    def required_multiplier(target_margin):
        target_profit = annual_operating_cost * target_margin / 100.0
        return (
            target_profit
            + annual_variable_cost
            + annual_operating_cost
            - annual_subsidy
        ) / annual_base_revenue

    lower = required_multiplier(MARGIN_LOWER)
    upper = required_multiplier(MARGIN_UPPER)
    return {
        'station_idx': station_idx,
        'raw_lower': float(lower),
        'raw_upper': float(upper),
        'clipped_lower': max(SEARCH_LOWER, float(lower)),
        'clipped_upper': min(SEARCH_UPPER, float(upper)),
        'annual_base_revenue': annual_base_revenue,
        'annual_variable_cost': annual_variable_cost,
        'annual_subsidy': annual_subsidy,
        'annual_operating_cost': annual_operating_cost,
    }


def build_station_multiplier_sets(active_stations, financial_config, t2_best, base_prices):
    coarse = set(rounded_range(SEARCH_LOWER, SEARCH_UPPER, COARSE_STEP))
    station_multiplier_sets = {}
    interval_estimates = {}

    for station_idx in active_stations:
        estimate = estimate_station_margin_interval(
            station_idx,
            financial_config,
            t2_best,
            base_prices,
        )
        interval_estimates[station_idx] = estimate
        values = set(coarse)

        if estimate is not None and estimate['clipped_lower'] <= estimate['clipped_upper']:
            local_start = max(SEARCH_LOWER, estimate['clipped_lower'] - LOCAL_PADDING)
            local_stop = min(SEARCH_UPPER, estimate['clipped_upper'] + LOCAL_PADDING)
            values.update(rounded_range(local_start, local_stop, FINE_STEP))
        elif estimate is not None:
            nearest = min(max((estimate['raw_lower'] + estimate['raw_upper']) / 2, SEARCH_LOWER), SEARCH_UPPER)
            local_start = max(SEARCH_LOWER, nearest - LOCAL_PADDING)
            local_stop = min(SEARCH_UPPER, nearest + LOCAL_PADDING)
            values.update(rounded_range(local_start, local_stop, FINE_STEP))

        station_multiplier_sets[station_idx] = sorted(values)

    return station_multiplier_sets, interval_estimates


def build_templates_for_used_multipliers(services, base_prices, station_multiplier_sets):
    all_multipliers = sorted({
        multiplier
        for values in station_multiplier_sets.values()
        for multiplier in values
    })
    templates = []
    multiplier_to_template_id = {}
    for multiplier in all_multipliers:
        family = 'coarse_grid' if math.isclose((multiplier - SEARCH_LOWER) % COARSE_STEP, 0.0, abs_tol=1e-9) else 'adaptive_fine_grid'
        template = package_station_template(
            len(templates),
            services,
            base_prices,
            multiplier,
            family,
        )
        templates.append(template)
        multiplier_to_template_id[multiplier] = template['template_id']
    return templates, multiplier_to_template_id


def build_candidate_plans(active_stations, station_multiplier_sets, templates_by_multiplier):
    plans = []
    station_value_lists = [station_multiplier_sets[station_idx] for station_idx in active_stations]
    for plan_id, multiplier_combo in enumerate(itertools.product(*station_value_lists)):
        station_template_ids = {}
        station_prices = {}
        price_plan_tuple = []
        for station_idx, multiplier in zip(active_stations, multiplier_combo):
            template = templates_by_multiplier[multiplier]
            station_template_ids[station_idx] = template['template_id']
            station_prices[station_idx] = dict(template['prices'])
            price_plan_tuple.extend(template['price_tuple'])

        plans.append({
            'plan_id': plan_id,
            'station_template_ids': station_template_ids,
            'station_uniform_multipliers': {
                station_idx: float(multiplier)
                for station_idx, multiplier in zip(active_stations, multiplier_combo)
            },
            'station_prices': station_prices,
            'price_plan_tuple': tuple(price_plan_tuple),
        })
    return plans


def main():
    print('========== T3_02: adaptive station pricing candidate generation ==========\n')
    financial_config, t2_best = load_inputs()
    active_stations = list(financial_config['active_stations'])
    paid_services = list(financial_config['paid_services'])
    base_prices = list(financial_config['service_prices_base'][:len(paid_services)])

    station_multiplier_sets, interval_estimates = build_station_multiplier_sets(
        active_stations,
        financial_config,
        t2_best,
        base_prices,
    )
    station_templates, multiplier_to_template_id = build_templates_for_used_multipliers(
        paid_services,
        base_prices,
        station_multiplier_sets,
    )
    templates_by_multiplier = {
        template['uniform_multiplier']: template
        for template in station_templates
    }
    candidate_plans = build_candidate_plans(
        active_stations,
        station_multiplier_sets,
        templates_by_multiplier,
    )

    print('[pricing rule]')
    print(f'  multiplier search range: [{SEARCH_LOWER:.2f}, {SEARCH_UPPER:.2f}]')
    print(f'  coarse step: {COARSE_STEP:.3f}')
    print(f'  local fine step: {FINE_STEP:.3f}')
    print('  emergency rescue remains free and is not a pricing variable.\n')

    print('[station local search windows]')
    for station_idx in active_stations:
        estimate = interval_estimates[station_idx]
        values = station_multiplier_sets[station_idx]
        if estimate is None:
            print(f'  station {station_idx}: no interval estimate, candidates={len(values)}')
            continue
        print(
            f"  station {station_idx}: estimated feasible multiplier "
            f"[{estimate['raw_lower']:.4f}, {estimate['raw_upper']:.4f}], "
            f"candidate count={len(values)}, range=[{min(values):.3f}, {max(values):.3f}]"
        )

    exhaustive_count = int(round(((SEARCH_UPPER - SEARCH_LOWER) / FINE_STEP + 1))) ** len(active_stations)
    print('\n[candidate scale]')
    print(f'  active stations: {len(active_stations)}')
    print(f'  station templates used: {len(station_templates):,}')
    print(f'  generated candidate plans: {len(candidate_plans):,}')
    print(f'  full 0.001 exhaustive station-uniform grid would be about: {exhaustive_count:,}')

    save_data = {
        'active_stations': active_stations,
        'paid_services': paid_services,
        'base_prices': base_prices,
        'search_lower': SEARCH_LOWER,
        'search_upper': SEARCH_UPPER,
        'coarse_step': COARSE_STEP,
        'fine_step': FINE_STEP,
        'local_padding': LOCAL_PADDING,
        'price_multipliers_by_station': station_multiplier_sets,
        'station_margin_interval_estimates': interval_estimates,
        'station_templates': station_templates,
        'candidate_plans': candidate_plans,
        'metadata': {
            'pricing_mode': 'adaptive_station_uniform_multipliers',
            'station_template_count': len(station_templates),
            'candidate_plan_count': len(candidate_plans),
            'full_fine_grid_count_estimate': exhaustive_count,
        },
    }

    os.makedirs('model', exist_ok=True)
    output_path = Path('model/T3_02_candidate_price_plans.pkl')
    with output_path.open('wb') as output_file:
        pickle.dump(save_data, output_file)

    print(f'\nSaved candidate plans to: {output_path}')
    print('================================================================\n')


if __name__ == '__main__':
    main()
