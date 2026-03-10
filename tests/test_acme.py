from openacme.icd10 import Icd10Graph
from openacme.acme import make_valid_range


def test_make_valid_range():
    g = Icd10Graph()
    # These are all real ranges that are problematic from the ACME tables
    assert make_valid_range('B59', 'B64', g) == ('B60', 'B64')
    assert make_valid_range('I84.0', 'I84.9', g) is None
    assert make_valid_range('D76.0', 'D86.9', g) == ('D76.1', 'D86.9')
    assert make_valid_range('M31.2', 'M31.3', g) == 'M31.3'
    assert make_valid_range('U01.0', 'U02', g) is None
    assert make_valid_range('A91', 'A92.4', g) == ('A92', 'A92.4')
    assert make_valid_range('B58.0', 'B59', g) == ('B58.0', 'B58.9')
