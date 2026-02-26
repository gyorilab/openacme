from openacme.acme import process_icd10_range, standardize_icd10


# --- standardize_icd10 ---


def test_standardize_letter_three_digits():
    assert standardize_icd10('A251') == 'A25.1'


def test_standardize_letter_four_digits():
    assert standardize_icd10('M1990') == 'M19.90'


def test_standardize_already_normalized():
    assert standardize_icd10('A25.1') == 'A25.1'


def test_standardize_category_only():
    assert standardize_icd10('C97') == 'C97'


# --- process_icd10_range ---


def test_process_range_single():
    assert process_icd10_range('R75') == {'code': 'R75', 'M': False, 'asterisk': False}


def test_process_range_range():
    assert process_icd10_range('A25.1-A25.9') == {
        'start': 'A25.1',
        'end': 'A25.9',
        'M': False,
        'asterisk': False,
    }


def test_process_range_with_M():
    r = process_icd10_range('M       A415')
    assert r['code'] == 'A41.5'
    assert r['M'] is True
    assert r['asterisk'] is False


def test_process_range_with_asterisk():
    r = process_icd10_range('Y560-Y569  *')
    assert r['start'] == 'Y56.0'
    assert r['end'] == 'Y56.9'
    assert r['asterisk'] is True
    assert r['M'] is False


def test_process_range_with_M_and_asterisk():
    r = process_icd10_range('M   R75 *')
    assert r['code'] == 'R75'
    assert r['M'] is True
    assert r['asterisk'] is True
