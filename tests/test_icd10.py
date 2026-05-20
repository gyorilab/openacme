from pathlib import Path

from openacme.icd10 import Icd10Graph, icd10_sort_key

RESOURCES = Path(__file__).parent / 'resources'
ICD10_FILE = str(RESOURCES / 'icd10_test.xml.zip')


def test_icd10_node_kinds():
    g = Icd10Graph(icd10_file=ICD10_FILE).graph
    assert g.nodes['I']['kind'] == 'chapter'
    assert g.nodes['A00-A09']['kind'] == 'block'
    assert g.nodes['A00.0']['kind'] == 'category'
    for n, data in g.nodes(data=True):
        assert data['kind'] in {'chapter', 'block', 'category'}


class TestIcd10:
    _icd10 = None

    @classmethod
    def icd10(cls):
        if cls._icd10 is None:
            cls._icd10 = Icd10Graph()
        return cls._icd10

    def test_expand_range1(self):
        codes = self.icd10().expand_icd10_range('Y43.1', 'Y43.4')
        expected_codes = ('Y43.1', 'Y43.2', 'Y43.3', 'Y43.4')
        assert codes == expected_codes

    def test_expand_range2(self):
        codes = self.icd10().expand_icd10_range('C00.0', 'C97')
        assert all(code.startswith('C') for code in codes)
        assert 'C00.0' in codes
        assert 'C97' in codes
        assert 'C50.9' in codes
        assert 'C34.1' in codes
        assert 'C98' not in codes

    def test_sort_key(self):
        assert icd10_sort_key('A00') < icd10_sort_key('B00')
        assert icd10_sort_key('C00') < icd10_sort_key('C01')
        assert icd10_sort_key('A00') < icd10_sort_key('A00.1')
        assert icd10_sort_key('Z99') < icd10_sort_key('U00')

    def test_valid_code(self):
        icd10 = self.icd10()
        assert icd10.find_next_valid_code('U01.0') == 'U04'
        # Valid codes just return themselves
        assert icd10.find_next_valid_code('A92') == 'A92'
        assert icd10.find_previous_valid_code('A92') == 'A92'
        # Non-existent codes return the next/previous valid code
        assert icd10.find_next_valid_code('A90') == 'A92'
        assert icd10.find_previous_valid_code('A90') == 'A89'

        assert icd10.find_next_valid_code('D75.2') == 'D75.8'
        assert icd10.find_previous_valid_code('D75.2') == 'D75.1'
