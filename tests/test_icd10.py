from openacme.icd10 import *


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


def test_sort_key():
    assert icd10_sort_key('A00') < icd10_sort_key('B00')
    assert icd10_sort_key('C00') < icd10_sort_key('C01')
    assert icd10_sort_key('A00') < icd10_sort_key('A00.1')
    assert icd10_sort_key('Z99') < icd10_sort_key('U00')


def test_valid_code():
    g = get_icd10_graph()
    regular_codes = get_regular_codes(g)
    assert find_next_valid_code(regular_codes, 'U01.0') == 'U04'
    # Valid codes just return themselves
    assert find_next_valid_code(regular_codes, 'A92') == 'A92'
    assert find_previous_valid_code(regular_codes, 'A92') == 'A92'
    # Non-existent codes return the next/previous valid code
    assert find_next_valid_code(regular_codes, 'A90') == 'A92'
    assert find_previous_valid_code(regular_codes, 'A90') == 'A89'

    assert find_next_valid_code(regular_codes, 'D75.2') == 'D75.8'
    assert find_previous_valid_code(regular_codes, 'D75.2') == 'D75.1'
