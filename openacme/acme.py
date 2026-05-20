"""This module implements processing ACME table D, which lists the ICD-10
codes that link codes based on possible underlying  causes of death.
The resulting graph can be used to expand the set of causes of death
associated with a given underlying cause of death, which is useful
for grounding clinical text to causes of death."""

import networkx as nx
from bs4 import BeautifulSoup
import re

from .icd10 import ICD10_BASE, icd10_sort_key, Icd10Graph

ACME_URL = "https://www.cdc.gov/nchs/nvss/manuals/2024/2c-2024-raw.html"


def standardize_icd10(raw_code, replacements):
    """Return standardized ICD10 codes from e.g., A251 to A25.1"""
    # If the code is 3 characters long, just return it. E.g. A25 -> A25
    if raw_code in replacements:
        return replacements[raw_code]
    elif len(raw_code) == 3:
        return raw_code
    # If the code is a letter followed by 3 numbers, we assume
    # that the last number should be separated by a .
    elif len(raw_code) == 4 and raw_code[0].isalpha() \
        and raw_code[1:4].isdigit():
        return f"{raw_code[0]}{raw_code[1:3]}.{raw_code[3]}"
    else:
        assert False, f"Unexpected ICD-10 code: {raw_code}"


def process_icd10_range(raw_range, replacements):
    """Process raw ICD10 range strings into tuples.

    Example: "A25.1-A25.9" -> ("A25.1", "A25.9")
    """
    # ('A25.1', 'A25.9')
    parts = raw_range.split('-')
    if len(parts) == 1:
        code = standardize_icd10(parts[0].strip(), replacements)
        return code
    elif len(parts) == 2:
        start = standardize_icd10(parts[0].strip(), replacements)
        end = standardize_icd10(parts[1].strip(), replacements)
        return start, end
    else:
        assert False, f"Unexpected ICD-10 range: {raw_range}"


def _find_table(soup, name):
    # <p class="H1" data-msection="Section_01" id="em_0010250">Table D<br /> ...
    for p in soup.find_all("p", class_="H1"):
        if "Table %s" % name in p.get_text(" ", strip=True):
            return p
    return None


def process_table_g(soup):
    """Return code replacements from ACME Table G.

    These represent custom codes that are introduced in ACME but are not real
    ICD10 codes. They are typically carve-outs of real ICD10 codes with
    additional exceptions. Later on, we could handle these more explicitly.
    For now, they are just mapped back to the original ICD10 code.
    """
    table_g_header = _find_table(soup, 'G')
    mappings = {}
    for tag in table_g_header.find_all_next('p'):
        classes = tag.get('class')
        # This is included in find_all_next
        if 'H2' in classes:
            continue
        # This is the start of Table F
        elif 'H1' in classes:
            break
        mapping_str = tag.get_text(strip=True)
        # There is an empty one of these
        if not mapping_str:
            break
        source, target = [part.strip()
                          for part in mapping_str.split(' ', maxsplit=1)]
        mappings[source] = standardize_icd10(target, replacements={})
    return mappings


def make_valid_range(range_start, range_end, icd10_graph):
    if range_start not in icd10_graph.graph:
        range_start = icd10_graph.find_next_valid_code(range_start)
    if range_end not in icd10_graph.graph:
        range_end = icd10_graph.find_previous_valid_code(range_end)
    # If the range collapses onto a single code, we just return that
    if range_start == range_end:
        return range_start
    # We have to handle the corner-case where the range is completely
    # out of the valid range in which case the new end will be before
    # the new start
    elif icd10_sort_key(range_start) > icd10_sort_key(range_end):
        return None
    return range_start, range_end


def process_table_d(icd10_graph, soup, replacements):
    """Return a graph representation of ACME relations from Table D.

    The graph will have nodes for both individual ICD-10 codes and ranges of
    codes, and edges from codes to their associated underlying causes of death.
    """
    table_d_header = _find_table(soup, 'D')
    parts = []
    # <p class="H2" data-msection="Section_01" id="em_0010251">A</p>
    current_h2 = None
    # <p class="H3" data-msection="Section_01" id="em_0010252">A000 Address</p>
    current_h3 = None
    # Skip these h3s since they are nonexistent in the ICD-10 reference
    skip_h3s = set()
    # Go until the next H1 or end
    for tag in table_d_header.find_all_next('p'):
        classes = set(tag.get('class') or [])
        # This would be the next table so we stop
        if 'H1' in classes and tag is not table_d_header:
            break
        elif 'H2' in classes:
            current_h2 = tag.get_text(" ", strip=True)
            continue
        # These are the "address" headings
        elif 'H3' in classes:
            current_h3 = standardize_icd10(
                tag.get_text(" ", strip=True).rstrip(' Address'),
                replacements=replacements
            )
            # If we are dealing with a code that is not in ICD-10 (removed or added
            # across versions, i.e., not one handled explicitly via replacements),
            # then we skip this part.
            if current_h3 not in icd10_graph.graph:
                skip_h3s.add(current_h3)
            continue
        # These are the rows under each address
        elif 'TableDRow' in classes:
            if current_h3 in skip_h3s:
                continue
            # Strip leading and trailing whitespace from the tag text.
            tag_text = tag.get_text(" ", strip=True)
            # Strip leading 'M' when it is followed by whitespace.
            # E.g. 'M   A25.1-A25.9' -> 'A25.1-A25.9'
            # Note: We could also add 'M' which means that the source "maybe"
            # the cause of the target, to the edge of 'causes' kind. 
            tag_text = re.sub(r'^M\s+', '', tag_text)
            # Strip trailing '*' when it is preceded by whitespace.
            # E.g. 'A25.1-A25.9  *' -> 'A25.1-A25.9'
            # Note: The trailing '*' likely is a revision marker from prior 
            # table, not a medically derived categorization.
            tag_text = re.sub(r'\s+\*$', '', tag_text)
            source = process_icd10_range(tag_text, replacements)
            # If we are dealing with a single code that isn't in ICD10, we
            # skip it
            if source not in icd10_graph.graph and \
                    not isinstance(source, tuple):
                continue
            # If we are dealing with a range, the beginning or end of which is
            # not in ICD10 then we adjust the range conservatively to find
            # the nearest valid code before/after the missing one (dependning
            # on which end of the interval we are on).
            if isinstance(source, tuple):
                source = make_valid_range(source[0], source[1], icd10_graph)
                # If the range could not be made valid
                if source is None:
                    continue
            parts.append({
                "block": current_h2,
                "target": current_h3,
                "source": source
            })
    nodes = {n: (n, {'kind': 'range' if isinstance(n, tuple) else 'category'})
             for n in ({part['target'] for part in parts} |
                       {part['source'] for part in parts})}
    edges = []
    expanded_ranges = set()
    for part in parts:
        edge = (part['source'], part['target'], {'kind': 'causes'})
        edges.append(edge)
        if isinstance(part['source'], tuple) and \
                part['source'] not in expanded_ranges:
            # Expand range
            codes_in_range = icd10_graph.expand_icd10_range(
                part['source'][0], part['source'][1]
            )
            for code in codes_in_range:
                # Add nodes that were only created via expansion
                if code not in nodes:
                    nodes[code] = (code, {'kind': 'category'})
                edges.append((code, part['source'],
                              {'kind': 'part_of_range'}))
            expanded_ranges.add(part['source'])
    g = nx.DiGraph()
    g.add_nodes_from(nodes.values())
    g.add_edges_from(edges)
    return g


def get_acme_graph(acme_file=None, icd10_file=None):
    """Return a graph representation of ACME Table D."""
    # If a file is provided, we use it. Otherwise, we download the HTML and cache it.
    if acme_file is None:
        # This downloads the HTML if not already there or uses a cached version.
        acme_file = ICD10_BASE.ensure(url=ACME_URL)

    # Parse the HTML and instantiate the soup object
    with open(acme_file, 'r') as fh:
        acme_text = fh.read()
    soup = BeautifulSoup(acme_text, features='lxml')

    # Get the base ICD-10 graph to use for expanding ranges
    g = Icd10Graph(icd10_file=icd10_file)

    # Process Table G to get replacements used in ACME e.g., E0390 and the
    # corresponding actual ICD10 code e.g., E03.9.
    replacements = process_table_g(soup)
    # Process Table D to get the ACME graph
    acme_g = process_table_d(g, soup, replacements)
    return acme_g


if __name__ == '__main__':
    g = get_acme_graph()


