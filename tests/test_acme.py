from openacme.icd10 import get_icd10_graph, get_regular_codes
from openacme.acme import make_valid_range


def test_make_valid_range():
    g = get_icd10_graph()
    regular_codes = get_regular_codes(g)
    # These are all real ranges that are problematic from the ACME tables
    assert make_valid_range('B59', 'B64', regular_codes, g) == ('B60', 'B64')
    assert make_valid_range('I84.0', 'I84.9', regular_codes, g) is None
    assert make_valid_range('D76.0', 'D86.9', regular_codes, g) \
        == ('D76.1', 'D86.9')
    assert make_valid_range('M31.2', 'M31.3', regular_codes, g) == 'M31.3'
    assert make_valid_range('U01.0', 'U02', regular_codes, g) is None
    assert make_valid_range('A91', 'A92.4', regular_codes, g) == ('A92', 'A92.4')
    assert make_valid_range('B58.0', 'B59', regular_codes, g) == ('B58.0', 'B58.9')
