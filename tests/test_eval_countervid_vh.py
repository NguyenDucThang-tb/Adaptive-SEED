import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts' / 'eval_countervid_vh.py'
spec = importlib.util.spec_from_file_location('eval_countervid_vh', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_loading_and_metrics(tmp_path):
    folder = tmp_path / 'temporal'
    folder.mkdir()
    (folder / 'v.mp4').touch()
    pair = {'basic': {'video': 'v.mp4', 'question': 'a?', 'answer': 'yes'},
            'hallucination': {'video': 'v.mp4', 'question': 'b?', 'answer': 'no'}}
    (folder / 'temporal.json').write_text(json.dumps([pair, pair]))
    rows = module.read_pairs(tmp_path, ['TPH'])
    assert len(rows) == 2
    for row in rows:
        row['base_prediction'] = row['answer']
        row['adaptive_prediction'] = 'yes'
    result = module.summarize(rows)['TPH']
    assert result['base']['pair_accuracy'] == 1
    assert result['adaptive']['accuracy'] == 0.5
    assert result['adaptive']['pair_accuracy'] == 0
    base_only = module.summarize(rows, ('base',))['TPH']
    assert 'adaptive' not in base_only
    assert base_only['base']['pair_accuracy'] == 1
    with pytest.raises(ValueError):
        module.summarize(rows[:1])
