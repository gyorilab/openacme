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


def test_standardize_with_decimal_unchanged():
    """Codes with decimal already in place pass through."""
    assert standardize_icd10('A25.10') == 'A25.10'


def test_standardize_non_matching_unchanged():
    """Codes that don't match 4 or 5 digit pattern pass through."""
    assert standardize_icd10('A2') == 'A2'
    assert standardize_icd10('A12345') == 'A12345'  # 6 chars, no match


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


def test_process_range_M_block_not_qualifier():
    """M-block codes (M02.1, M02.1-M02.9) are ICD-10 codes, not M qualifier."""
    r = process_icd10_range('M02.1-M02.9')
    assert r['start'] == 'M02.1'
    assert r['end'] == 'M02.9'
    assert r['M'] is False


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


def test_process_range_M_with_range():
    """M qualifier with a range."""
    r = process_icd10_range('M A25.1-A25.9')
    assert r['start'] == 'A25.1'
    assert r['end'] == 'A25.9'
    assert r['M'] is True
    assert r['asterisk'] is False


def test_process_range_M_single_space():
    """M with single space after."""
    r = process_icd10_range('M R75')
    assert r['code'] == 'R75'
    assert r['M'] is True


def test_process_range_asterisk_only():
    """Asterisk without M."""
    r = process_icd10_range('Y560-Y569*')
    assert r['start'] == 'Y56.0'
    assert r['end'] == 'Y56.9'
    assert r['asterisk'] is True
    assert r['M'] is False
