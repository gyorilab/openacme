"""This module implements processing ACME table D, which lists the ICD-10
codes that link codes based on possible underlying  causes of death.
The resulting graph can be used to expand the set of causes of death
associated with a given underlying cause of death, which is useful
for grounding clinical text to causes of death."""

import networkx as nx
import tqdm
from bs4 import BeautifulSoup

from .icd10 import ICD10_BASE, expand_icd10_range, get_icd10_graph

ACME_URL = "https://www.cdc.gov/nchs/nvss/manuals/2024/2c-2024-raw.html"


def standardize_icd10(raw_code):
    """Return standardized ICD10 codes from e.g., A251 to A25.1"""
    # If the code is a letter followed by 3 numbers, we assume
    # that the last number should be separated by a .
    if len(raw_code) == 4 and raw_code[0].isalpha() and raw_code[1:4].isdigit():
        return f"{raw_code[0]}{raw_code[1:3]}.{raw_code[3]}"
    # If the code is a letter followed by 4 numbers, we assume
    # that the last two numbers should be separated by a .
    if len(raw_code) == 5 and raw_code[0].isalpha() and raw_code[1:5].isdigit():
        return f"{raw_code[0]}{raw_code[1:3]}.{raw_code[3:]}"
    return raw_code


def process_icd10_range(raw_range):
    """Process raw ICD10 range strings. Returns dict with code or start/end plus M, asterisk."""
    # We assume that a range can start with an M and/or end with an asterisk
    s = raw_range.strip()
    has_M = s.startswith('M')
    has_asterisk = s.endswith('*')
    s = s.replace('*', '').strip()
    if has_M:
        s = s[1:].lstrip()
    # Return a dictionary with the code or start/end plus M, asterisk
    parts = s.split('-')
    if len(parts) == 1:
        code = standardize_icd10(parts[0].strip())
        return {"code": code, "M": has_M, "asterisk": has_asterisk}
    elif len(parts) == 2:
        start = standardize_icd10(parts[0].strip())
        end = standardize_icd10(parts[1].strip())
        return {"start": start, "end": end, "M": has_M, "asterisk": has_asterisk}
    else:
        assert False, f"Unexpected ICD-10 range: {raw_range}"


def process_table_d(icd10_graph, soup):
    """Return a graph representation of ACME relations from Table D.

    The graph will have nodes for both individual ICD-10 codes and ranges of
    codes, and edges from codes to their associated underlying causes of death.
    """
    # Find the TableD section
    # <p class="H1" data-msection="Section_01" id="em_0010250">Table D<br /> ...
    table_d_header = None
    for p in soup.find_all("p", class_="H1"):
        if "Table D" in p.get_text(" ", strip=True):
            table_d_header = p
            break
    if not table_d_header:
        return

    parts = []
    # <p class="H2" data-msection="Section_01" id="em_0010251">A</p>
    current_h2 = None
    # <p class="H3" data-msection="Section_01" id="em_0010252">A000 Address</p>
    current_h3 = None
    # Go until the next H1 or end
    for tag in tqdm.tqdm(table_d_header.find_all_next('p')):
        classes = set(tag.get('class') or [])
        # This would be the next table so we stop
        if 'H1' in classes and tag is not table_d_header:
            break
        elif 'H2' in classes:
            current_h2 = tag.get_text(" ", strip=True)
            continue
        elif 'H3' in classes:
            current_h3 = standardize_icd10(
                tag.get_text(" ", strip=True).rstrip(' Address')
            )
            continue
        elif 'TableDRow' in classes:
            source = process_icd10_range(tag.get_text(" ", strip=True))
            parts.append({
                "block": current_h2,
                "target": current_h3,
                "source": source
            })

    def _source_node(src):
        return src['code'] if 'code' in src else (src['start'], src['end'])

    nodes = [(n, {'type': 'range' if isinstance(n, tuple) else 'code'})
             for n in ({part['target'] for part in parts} |
                       {_source_node(part['source']) for part in parts})]
    edges = []
    for part in parts:
        src = part['source']
        node = _source_node(src)
        edge_attrs = {'kind': 'causes', 'M': src.get('M', False), 'asterisk': src.get('asterisk', False)}
        edges.append((node, part['target'], edge_attrs))
        if 'start' in src and 'end' in src:
            codes_in_range = expand_icd10_range(icd10_graph, src['start'], src['end'])
            for code in codes_in_range:
                edges.append((code, node, {'kind': 'part_of_range'}))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def get_acme_graph():
    """Return a graph representation of ACME Table D."""
    # This downloads the HTML if not already there or uses a cached version.
    acme_file = ICD10_BASE.ensure(url=ACME_URL)

    # Parse the HTML and instantiate the soup object
    with open(acme_file, 'r') as fh:
        acme_text = fh.read()
    soup = BeautifulSoup(acme_text, features='lxml')

    # Get the base ICD-10 graph to use for expanding ranges
    g = get_icd10_graph()

    # Process Table D to get the ACME graph
    acme_g = process_table_d(g, soup)
    return acme_g


if __name__ == '__main__':
    g = get_acme_graph()


