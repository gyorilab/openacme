import networkx as nx

from openacme.icd10 import expand_icd10_range, get_icd10_graph


def test_expand_range1():
    g = get_icd10_graph()
    codes = expand_icd10_range(g, 'Y43.1', 'Y43.4')
    expected_codes = ['Y43.1', 'Y43.2', 'Y43.3', 'Y43.4']
    assert codes == expected_codes


def test_expand_range2():
    g = get_icd10_graph()
    codes = expand_icd10_range(g, 'C00.0', 'C97')
    assert all(code.startswith('C') for code in codes)
    assert 'C00.0' in codes
    assert 'C97' in codes
    assert 'C50.9' in codes
    assert 'C34.1' in codes
    assert 'C98' not in codes
    assert 'C97-C97' not in codes  # blocks excluded


def test_expand_range_decimal_ordering():
    """Sort uses (prefix, sub as int) so A00.9 < A00.10 (numeric, not lexicographic)."""
    # Use a minimal graph with just these codes for this test
    g = nx.DiGraph()
    for code in ['A00.0', 'A00.1', 'A00.9', 'A00.10']:
        g.add_node(code, type='code')
    codes = expand_icd10_range(g, 'A00.0', 'A00.10')
    assert 'A00.9' in codes
    assert 'A00.10' in codes
    assert codes.index('A00.9') < codes.index('A00.10')
