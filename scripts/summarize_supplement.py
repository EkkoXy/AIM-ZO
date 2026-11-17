"""Validate released supplementary tables and reproduce their aggregates on CPU."""
import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def load_csv(path):
    return list(csv.DictReader((ROOT / path).open()))


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def aggregate(rows, keys, value):
    groups = defaultdict(list)
    for row in rows:
        score = float(row[value])
        assert math.isfinite(score)
        groups[tuple(row[k] for k in keys)].append(score)
    return [{**dict(zip(keys, group)), "n": len(scores), "mean": mean(scores),
             "sample_sd": stdev(scores) if len(scores) > 1 else 0.0}
            for group, scores in sorted(groups.items())]


def build():
    outputs = {}
    alignment = load_csv("results/diagnostics/activation-gradient/per-module.csv")
    assert len(alignment) == 612
    assert len({(r['task'], r['checkpoint_step'], r['module']) for r in alignment}) == 612
    coverage = []
    for task, modules in [("rte", 192), ("sst2", 12)]:
        for step in [500, 1500, 2500]:
            path = ROOT / f"results/diagnostics/activation-gradient/{task}-step{step}.json"
            d = json.loads(path.read_text())
            assert not set(d['basis_indices']) & set(d['gradient_indices'])
            assert len(d['basis_indices']) == len(d['gradient_indices']) == 32
            assert d['training_steps'] == 0 and len(d['rows']) == modules
            selected = [r for r in alignment if r['task'] == task and int(r['checkpoint_step']) == step]
            assert len(selected) == modules
            for row, raw in zip(selected, d['rows']):
                assert row['module'] == raw['module']
                assert float(row['activation_coverage']) == raw['activation_coverage']
                assert 0 <= float(row['activation_coverage']) <= 1.00001
                assert math.isclose(float(row['excess_coverage']),
                    float(row['activation_coverage']) - mean(float(row[f'random_coverage_{i}']) for i in range(1, 6)),
                    abs_tol=1e-7)
            coverage.append(dict(task=task, checkpoint_step=step, modules=modules,
                activation_coverage_percent=100*mean(float(r['activation_coverage']) for r in selected),
                random_coverage_percent=100*mean(float(r[f'random_coverage_{i}']) for r in selected for i in range(1, 6))))
    outputs['results/diagnostics/activation-gradient/summary.csv'] = csv_text(coverage)
    for folder, keys, count in [('active-width', ['k'], 6), ('shared-width', ['series'], 10)]:
        rows = load_csv(f'results/ablations/{folder}/per-seed.csv')
        assert len(rows) == count
        assert len({tuple(r[k] for k in keys) + (r['seed'],) for r in rows}) == count
        outputs[f'results/ablations/{folder}/summary.csv'] = csv_text(aggregate(rows, keys, 'accuracy_percent'))
    for method in ['agzo', 'lozo']:
        d = json.loads((ROOT / f'results/audits/baseline-parity/{method}.json').read_text())
        assert d['status'] == 'PASS' and not d['mismatches']
        assert d['full_initial_final_hashes_equal']
        assert d['arms'][0]['initial_model_sha256'] == d['arms'][1]['initial_model_sha256']
        assert d['arms'][0]['final_model_sha256'] == d['arms'][1]['final_model_sha256']
    kernels = json.loads((ROOT / 'results/audits/baseline-parity/lozo-curvzo-kernels.json').read_text())
    for method, steps in [('lozo', 7), ('curvzo', 60)]:
        assert len(kernels[method]) == 3
        for run in kernels[method]:
            assert len(run['rows']) == steps
            assert all(r['perturb_equal'] and r['update_equal'] for r in run['rows'])
    precision = load_csv('results/diagnostics/precision/per-seed.csv')
    assert len(precision) == 8
    assert len({(r['method'], r['dtype'], r['seed']) for r in precision}) == 8
    assert len({r['official_payload_sha256'] for r in precision}) == 1
    outputs['results/diagnostics/precision/summary.csv'] = csv_text(aggregate(precision, ['method', 'dtype'], 'f1_percent'))
    assert len(load_csv('results/tuning/candidate-parameters.csv')) == 256
    tuning = load_csv('results/tuning/agzo-squad-per-run.csv')
    assert len(tuning) == 185 and len({r['pair_id'] for r in tuning}) == 65
    folder = 'experiments/appendix/lora-selected-layers'
    runs = load_csv(f'{folder}/results/per-seed.csv')
    assert len(runs) == 66
    assert len({(r['task'], r['method'], r['seed']) for r in runs}) == 66
    dev = {r['run_id']: r['evaluations'] for r in json.loads((ROOT / f'{folder}/results/development.json').read_text())}
    groups = defaultdict(list)
    for r in runs:
        assert int(r['calls']) == 40000
        assert int(r['steps']) == (20000 if r['method'] == 'MeZO+LoRA' else 2500)
        evaluations = dev[r['run_id']]
        assert len(evaluations) == 5
        best = min(evaluations, key=lambda e: (e['loss'], e['step']))
        assert int(r['best_step']) == best['step']
        assert float(r['best_dev_loss']) == best['loss']
        groups[r['task']].append(r)
    for group in groups.values():
        assert len(group) == 6
        assert len({r['payload_sha256'] for r in group}) == 1
        for method in ['MeZO+LoRA', 'AIM-ZO']:
            assert {int(r['seed']) for r in group if r['method'] == method} == {42, 142, 242}
    percent = [{**r, 'score_percent': 100*float(r['score'])} for r in runs]
    stats = aggregate(percent, ['task', 'method', 'metric'], 'score_percent')
    outputs[f'{folder}/results/summary.csv'] = csv_text(stats)
    lookup = {(r['task'], r['method']): r for r in stats}
    text = ['# Results', '', 'Mean +/- sample standard deviation across seeds 42, 142, and 242. Scores are percentages.', '',
            '| Task | Metric | MeZO+LoRA | AIM-ZO |', '| --- | --- | ---: | ---: |']
    labels = {'rte': 'RTE', 'sst2': 'SST-2', 'boolq': 'BoolQ', 'wic': 'WiC', 'cb': 'CB', 'copa': 'COPA',
              'wsc': 'WSC', 'squad': 'SQuAD', 'multirc': 'MultiRC', 'record': 'ReCoRD', 'drop': 'DROP'}
    for task, label in labels.items():
        a, b = lookup[(task, 'MeZO+LoRA')], lookup[(task, 'AIM-ZO')]
        text.append(f"| {label} | {a['metric']} | {a['mean']:.2f} +/- {a['sample_sd']:.2f} | {b['mean']:.2f} +/- {b['sample_sd']:.2f} |")
    outputs[f'{folder}/RESULTS.md'] = '\n'.join(text) + '\n'
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check committed aggregates without changing files.')
    args = parser.parse_args()
    outputs = build()
    for name, content in outputs.items():
        path = ROOT / name
        if args.check:
            assert path.read_text() == content, f'Aggregate differs: {name}'
        else:
            path.write_text(content)
    print('PASS: alignment, width, parity, precision, tuning, and 66 paired task runs; zero model calls')


if __name__ == '__main__':
    main()
