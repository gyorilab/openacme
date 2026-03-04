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
           'expand_icd10_range']

import zipfile
from lxml import etree
from collections import defaultdict
import networkx as nx
import re

from .. import OPENACME_BASE

ICD10_BASE = OPENACME_BASE.module('icd10')
ICD10_XML_URL = "https://icdcdn.who.int/icd10/claml/icd102019en.xml.zip"


def expand_icd10_range(g, start, end):
    # Return a list of all codes between e.g.,
    # ('Y43.1', 'Y43.4') or ('C00.0', 'C97')
    codes = sorted(g.nodes)
    in_range = []
    in_range_flag = False
    end_is_super_class = ('.' not in end)
    for code in codes:
        if code == start:
            in_range_flag = True
        if in_range_flag:
            in_range.append(code)
        if code == end or (end_is_super_class and code.startswith(end)):
            break
    return in_range


def get_icd10_graph():
    zip_path = ICD10_BASE.ensure(url=ICD10_XML_URL)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        xml_name = zf.namelist()[0]
        with zf.open(xml_name) as fh:
            tree = etree.parse(fh)

    # Preprocess modifiers that we can later reference
    modifier_tags = tree.findall('Modifier')
    modifier_class_tags = tree.findall('ModifierClass')
    modifier_classes = defaultdict(dict)
    # We differentiate codes that end in _4 and _5 because
    # _4-level modifiers are used as part of ICD10 codes whereas
    # _5-level modifiers are supposed to be represented separately from codes.
    modifier_types = {}
    for m in modifier_class_tags:
        # Note that level 4 codes include a prepended . like .9 whereas
        # level 5 codes do not contain this period prefix
        code = m.attrib['code']
        modifier = m.attrib['modifier']
        # Note: there is also a "kind" attribute for the Rubric
        # that we may want to pick up
        label = m.find('Rubric/Label').text
        modifier_classes[modifier][code] = label
        if modifier.endswith('_4'):
            modifier_types[modifier] = '4_level'
        else:
            modifier_types[modifier] = '5_level'
    modifier_classes = dict(modifier_classes)

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
        rubric_data = dict(rubric_data)
        nodes.append([code, {'kind': kind, 'rubrics': rubric_data}])
        # Some categories reference their subdivisions via ModifiedBy; those subdivisions
        # live in separate Modifier elements rather than as Class elements. Add them.
        modified_by = cls.find('ModifiedBy')
        if modified_by is not None:
            modifier_code = modified_by.attrib['code']
            if kind == 'category' and modifier_types[modifier_code] == '4_level':
                for subclass_code, subclass_label in modifier_classes[modifier_code].items():
                    preferred_names = []
                    for name in rubric_data['preferred']:
                        # The XML defines the subclasses with first letter capitalized
                        # but on icd10.who.int they are lowercase so we use that convention here
                        # when generating the names
                        preferred_names.append(name + ' : ' + subclass_label.lower())
                    subclass_rubric_data = {'preferred': preferred_names}
                    full_code = code + subclass_code
                    nodes.append([full_code, {'kind': 'category', 'rubrics': subclass_rubric_data}])
                    edges.append((full_code, code, {'kind': 'is_a'}))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    g = get_icd10_graph()
    print(f'ICD-10 graph has {len(g.nodes)} nodes and {len(g.edges)} edges.')
