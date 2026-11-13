"""Compare complete Video-MME evaluations with an identically scored Base."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from seed_video.evaluation.videomme import compare, summarize
from eval_videomme import sha


def read_complete(directory):
    marker, predictions = directory / 'COMPLETE.json', directory / 'predictions.json'
    if not marker.exists() or not predictions.exists():
        return None
    completion = json.loads(marker.read_text())
    payload = json.loads(predictions.read_text())
    if completion['predictions_sha'] != sha(predictions) or completion['n'] != len(payload['predictions']):
        raise ValueError(f'Completion marker does not match predictions: {directory}')
    return payload, completion


def collect(root, epsilon_pp=None, iterations=2000):
    config_path = root / 'experiment.json'
    declared_margin = json.loads(config_path.read_text()).get('epsilon_pp', 1.0) if config_path.exists() else 1.0
    if epsilon_pp is not None and config_path.exists() and epsilon_pp != declared_margin:
        raise ValueError('Non-inferiority margin differs from the predeclared experiment.json value.')
    epsilon_pp = declared_margin if epsilon_pp is None else epsilon_pp
    base = read_complete(root / 'evaluation/base')
    if not base:
        raise ValueError('Base Video-MME evaluation is not complete yet.')
    base_payload, base_marker = base
    base_rows = base_payload['predictions']
    base_scores = {(r['group'], r['value']): r for r in summarize(base_rows)}
    results, details = [], []
    for name in ('base', 'videohallucer', 'eventhallusion', 'mixed'):
        current = read_complete(root / 'evaluation' / name)
        if current is None:
            results.append(dict(train_source=name, status='INCOMPLETE'))
            continue
        payload, marker = current
        if payload['meta']['protocol'] != base_payload['meta']['protocol']:
            raise ValueError(f'Base/{name} evaluation protocols differ; do not compare.')
        smoke = marker['smoke'] or base_marker['smoke']
        scores = {(r['group'], r['value']): r for r in summarize(payload['predictions'])}
        row = dict(train_source=name, status='SMOKE_COMPLETE' if smoke else 'RESULTS_READY',
                   n=len(payload['predictions']), invalid=scores['overall', 'all']['invalid'])
        for label, group, value in [('overall', 'overall', 'all'), ('short', 'duration', 'short'),
                                    ('medium', 'duration', 'medium'), ('long', 'duration', 'long')]:
            row[label] = scores.get((group, value), {}).get('accuracy', '')
        if not smoke:
            row.update(compare(base_rows, payload['predictions'], epsilon_pp, iterations))
        for key, score in scores.items():
            reference = base_scores[key]
            details.append(dict(train_source=name, status=row['status'], **score,
                                base_accuracy=reference['accuracy'],
                                delta_pp=100 * (score['accuracy'] - reference['accuracy'])))
        results.append(row)
    for name, rows in [('retention_results.csv', results), ('retention_details.csv', details)]:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (root / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print('Saved:', root / name)
    print(f'{"Source":<16} {"Status":<16} {"Overall%":>9} {"Short%":>9} {"Medium%":>9} {"Long%":>9} {"Delta pp":>9}')
    for r in results:
        values = [f'{100 * r[k]:.2f}' if r.get(k) != '' and k in r else '-' for k in ('overall', 'short', 'medium', 'long')]
        delta = f'{r["delta_pp"]:+.2f}' if 'delta_pp' in r else '-'
        print(f'{r["train_source"]:<16} {r["status"]:<16} ' + ' '.join(f'{v:>9}' for v in values) + f' {delta:>9}')
    print('RESULTS_READY validates artifacts, not PBS exit status. Smoke runs cannot support retention conclusions.')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--epsilon-pp', type=float, help='Must match experiment.json; otherwise inherit its predeclared margin.')
    parser.add_argument('--bootstrap-iterations', type=int, default=2000)
    args = parser.parse_args()
    collect(args.root, args.epsilon_pp, args.bootstrap_iterations)


if __name__ == '__main__':
    main()
