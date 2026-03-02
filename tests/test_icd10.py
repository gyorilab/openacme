import networkx as nx

from openacme.icd10 import (
    convert_icd10_code_to_range,
    expand_icd10_range,
    get_icd10_graph,
    sort_icd10_graph_codes_and_blocks,
)


# --- convert_icd10_code_to_range ---


def test_convert_single_code():
    """Point code maps to same start and end."""
    r = convert_icd10_code_to_range('A12.10')
    assert r == ('A', 12, 10, 'A', 12, 10)


def test_convert_category_no_decimal():
    """Category without decimal (e.g. C97) spans to inf."""
    r = convert_icd10_code_to_range('C97')
    assert r == ('C', 97, 0, 'C', 97, float('inf'))


def test_convert_block():
    """Block range parses start and end."""
    r = convert_icd10_code_to_range('C97-C98')
    assert r == ('C', 97, 0, 'C', 98, float('inf'))


def test_convert_chapter():
    """Single letter chapter."""
    r = convert_icd10_code_to_range('C')
    assert r == ('C', 0, 0, 'C', float('inf'), float('inf'))


def test_convert_invalid_raises():
    """Invalid format raises ValueError."""
    try:
        convert_icd10_code_to_range('021')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Cannot parse" in str(e)


# --- expand_icd10_range ---


def test_expand_range1():
    g = get_icd10_graph()
    sorted_codes_and_blocks = sort_icd10_graph_codes_and_blocks(g)
    codes = expand_icd10_range(sorted_codes_and_blocks, 'Y43.1', 'Y43.4')
    expected_codes = ['Y43.1', 'Y43.2', 'Y43.3', 'Y43.4']
    assert codes == expected_codes


def test_expand_range2():
    g = get_icd10_graph()
    sorted_codes_and_blocks = sort_icd10_graph_codes_and_blocks(g)
    codes_and_blocks = expand_icd10_range(sorted_codes_and_blocks, 'C00.0', 'C97')
    assert all(item.startswith('C') for item in codes_and_blocks)
    assert 'C00.0' in codes_and_blocks
    assert 'C97' in codes_and_blocks
    assert 'C50.9' in codes_and_blocks
    assert 'C34.1' in codes_and_blocks
    assert 'C98' not in codes_and_blocks


def test_expand_range_decimal_ordering():
    """Sort uses (prefix, sub as int) so A00.9 < A00.10 (numeric, not lexicographic)."""
    g = nx.DiGraph()
    for code in ['A00.0', 'A00.1', 'A00.9', 'A00.10']:
        g.add_node(code, type='code')
    sorted_codes_and_blocks = sort_icd10_graph_codes_and_blocks(g)
    codes = expand_icd10_range(sorted_codes_and_blocks, 'A00.0', 'A00.10')
    assert 'A00.9' in codes
    assert 'A00.10' in codes
    assert codes.index('A00.9') < codes.index('A00.10')


def test_expand_range_includes_blocks():
    """Blocks in range are included."""
    g = nx.DiGraph()
    g.add_node('C00.0', type='code')
    g.add_node('C00-C14', type='block')
    g.add_node('C15-C26', type='block')
    g.add_node('C97', type='code')
    sorted_codes_and_blocks = sort_icd10_graph_codes_and_blocks(g)
    result = expand_icd10_range(sorted_codes_and_blocks, 'C00.0', 'C97')
    assert 'C00.0' in result
    assert 'C00-C14' in result
    assert 'C15-C26' in result
    assert 'C97' in result


def test_expand_range_single_code():
    """Range to single code returns just that code."""
    g = nx.DiGraph()
    g.add_node('A25.1', type='code')
    sorted_codes_and_blocks = sort_icd10_graph_codes_and_blocks(g)
    result = expand_icd10_range(sorted_codes_and_blocks, 'A25.1', 'A25.1')
    assert result == ['A25.1']


def test_expand_range_empty_when_no_overlap():
    """Empty when query range has no overlap with graph."""
    g = nx.DiGraph()
    for code in ['A00.0', 'A00.1']:
        g.add_node(code, type='code')
    sorted_codes_and_blocks = sort_icd10_graph_codes_and_blocks(g)
    result = expand_icd10_range(sorted_codes_and_blocks, 'Z99.0', 'Z99.9')
    assert result == []


# --- sort_icd10_graph_codes_and_blocks ---


def test_sort_orders_by_range_start():
    """Sort orders by (letter, maj, min) of range start."""
    g = nx.DiGraph()
    for node in ['A02.1', 'A00.9', 'A01.0', 'A00.0']:
        g.add_node(node, type='code')
    sorted_cb = sort_icd10_graph_codes_and_blocks(g)
    codes = [c[0] for c in sorted_cb]
    assert codes == ['A00.0', 'A00.9', 'A01.0', 'A02.1']
