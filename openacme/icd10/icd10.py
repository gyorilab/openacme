"""
Download page: https://icdcdn.who.int/icd10/index.html
there are two versions: a zipped XML file or a zipped set of three
tabular text files. The text files are documented here:
https://icdcdn.who.int/icd10/metainfo.html and the XML
here: https://icdcdn.who.int/icd10/clamlinfo.html.

The text files have the key structure but don't contain as much
additional detail as the XML, notably inclusion and exclusion
criteria seem to be included only in the XML.

The structure of ICD-10 consists of chapters (roman numerals),
under which are blocks (alphanumeric ranges), and under blocks
are categories (alphanumeric codes) and there are sub-categories
with decimal points. For example:

Chapter (I) -> Block (A00-A09) -> Category (A00) -> Category (A00.0)
"""
__all__ = ['ICD10_BASE', 'ICD10_XML_URL', 'get_icd10_graph',
           'convert_icd10_code_to_range', 'expand_icd10_range']

import re
import zipfile
from lxml import etree
from collections import defaultdict
import networkx as nx

from .. import OPENACME_BASE

ICD10_BASE = OPENACME_BASE.module('icd10')
ICD10_XML_URL = "https://icdcdn.who.int/icd10/claml/icd102019en.xml.zip"


def convert_icd10_code_to_range(s):
    """Convert ICD-10 code/block/chapter to (letter, maj, min, letter, maj, min).
    Use inf for unbounded end. Examples:
    - A12.10 (code) -> (A, 12, 10, A, 12, 10)
    - C97 (block) -> (C, 97, 0, C, 97, inf)
    - C97-C98 (range) -> (C, 97, 0, C, 98, inf)
    - C (chapter) -> (C, 0, 0, C, inf, inf)
    """

    def _parse_part(s, *, for_end=False):
        """Parse single code to (letter, category, subcategory). for_end=True uses inf for unbounded."""
        if len(s) == 1 and s.isalpha():
            return (s.upper(), 0, 0) if not for_end else (s.upper(), float('inf'), float('inf'))
        m = re.match(r'^([A-Z])(\d{2,3})(?:\.(\d+))?$', s, re.I)
        if not m:
            raise ValueError(f"Cannot parse ICD-10 code: {s}")
        letter, cat, subcat = m.group(1).upper(), int(m.group(2)), m.group(3)
        subcat_val = int(subcat) if subcat else (float('inf') if for_end else 0)
        return (letter, cat, subcat_val)

    s = s.strip()
    if '-' in s and not s.startswith('-'):
        start_s, end_s = s.split('-', 1)
        start_s, end_s = start_s.strip(), end_s.strip()
        return (*_parse_part(start_s, for_end=False), *_parse_part(end_s, for_end=True))
    return (*_parse_part(s, for_end=False), *_parse_part(s, for_end=True))


def expand_icd10_range(icd10_graph, start, end):
    """Return codes in [start, end]. Uses convert_icd10_code_to_range for comparison."""
    query_r = convert_icd10_code_to_range(f"{start}-{end}")
    query_start = query_r[:3]
    query_end = query_r[3:]

    codes_in_range = []
    for code in icd10_graph.nodes:
        if icd10_graph.nodes[code].get('type') != 'code':
            continue
        code_r = convert_icd10_code_to_range(code)
        code_start = code_r[:3]
        code_end = code_r[3:]
        if query_start <= code_start and code_end <= query_end:
            codes_in_range.append(code)
    return codes_in_range

def get_icd10_graph():
    zip_path = ICD10_BASE.ensure(url=ICD10_XML_URL)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        xml_name = zf.namelist()[0]
        with zf.open(xml_name) as fh:
            tree = etree.parse(fh)

    # All terms are represented as <Class> elements
    classes = tree.findall('Class')
    nodes = []
    edges = []
    for cls in classes:
        code = cls.attrib['code']
        kind = cls.attrib['kind']
        # Establish is_a relationships from categories to blocks
        # and from blocks to chapters
        if kind in {'category', 'block'}:
            superclass = cls.find('SuperClass').attrib['code']
            edges.append((code, superclass, {'kind': 'is_a'}))
            assert superclass is not None
        # Extra data is available in rubrics, typically alternative
        # names or inclusion/exclusion criteria
        rubric_data = defaultdict(list)
        for rubric in cls.findall('Rubric'):
            rubric_kind = rubric.attrib['kind']
            name = rubric.find('Label').text
            name = name.strip() if name else None
            if name:
                rubric_data[rubric_kind].append(name)
        node_type = 'code' if kind == 'category' else ('block' if kind == 'block' else 'chapter')
        nodes.append([code, {'kind': kind, 'type': node_type, 'rubrics': dict(rubric_data)}])

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    g = get_icd10_graph()
    print(f'ICD-10 graph has {len(g.nodes)} nodes and {len(g.edges)} edges.')
