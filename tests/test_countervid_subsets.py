import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_countervid_subsets.py'
spec = importlib.util.spec_from_file_location('subsets', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nested_deterministic_and_order_independent():
    train = [{'id': str(i), 'base_identifier': str(i)} for i in range(20)]
    validation = [{'id': 'v', 'base_identifier': 'v'}]
    a = module.build_manifest(train, validation, [5, 10], 42)
    b = module.build_manifest(train[::-1], validation, [10, 5], 42)
    assert a == b
    assert a['subsets']['5'] == a['subsets']['10'][:5]


def test_overlap_and_oversized_subset_rejected():
    train = [{'id': '1', 'base_identifier': 'x'}]
    with pytest.raises(ValueError, match='overlap'):
        module.build_manifest(train, [{'id': '2', 'base_identifier': 'x'}], [1], 42)
    with pytest.raises(ValueError, match='Subset sizes'):
        module.build_manifest(train, [], [2], 42)
