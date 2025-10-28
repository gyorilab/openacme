"""
Download page: https://icdcdn.who.int/icd10/index.html
there are two versions: a zipped XML file or a zipped set of three
tabular text files. The text files are documented here:
https://icdcdn.who.int/icd10/metainfo.html and the XML
here: https://icdcdn.who.int/icd10/clamlinfo.html.

The text files have the key structure but don't contain as much
additional detail as the XML, notably inclusion and exclusion
criteria seem to be included only in the XML.

The structure of ICD-10 consistes of chapters (roman numberals),
under which are blocks (alphanumeric ranges), and under blocks
are categories (alphanumeric codes) and there are sub-categories
with decimal points. For example:

Chapter (I) -> Block (A00-A09) -> Category (A00) -> Category (A00.0)
"""
__all__ = ['ICD10_BASE', 'ICD10_XML_URL', 'get_icd10_graph']

import zipfile
from lxml import etree
from collections import defaultdict
import networkx as nx

from openacme import OPENACME_BASE

ICD10_BASE = OPENACME_BASE.module('icd10')
ICD10_XML_URL = "https://icdcdn.who.int/icd10/claml/icd102019en.xml.zip"


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
            rubric_data[rubric_kind].append(name)
        nodes.append([code, {'kind': kind, 'rubrics': dict(rubric_data)}])

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    g = get_icd10_graph()
    print(f'ICD-10 graph has {len(g.nodes)} nodes and {len(g.edges)} edges.')