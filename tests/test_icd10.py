from openacme.icd10 import get_icd10_graph, expand_icd10_range


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