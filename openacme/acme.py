import networkx as nx
import tqdm
from bs4 import BeautifulSoup

from .icd10 import ICD10_BASE, expand_icd10_range, get_icd10_graph

ACME_URL = "https://www.cdc.gov/nchs/nvss/manuals/2024/2c-2024-raw.html"


def standardize_icd10(raw_code):
    # If the code is a letter followed by 3 numbers, we assume
    # that the last number should be separated by a .
    if len(raw_code) == 4 and raw_code[0].isalpha() and raw_code[1:4].isdigit():
        return f"{raw_code[0]}{raw_code[1:3]}.{raw_code[3]}"
    return raw_code


def process_icd10_range(raw_range):
    # Process A251\xa0\xa0 -A259 into
    # ('A25.1', 'A25.9')
    parts = raw_range.split('-')
    if len(parts) == 1:
        code = standardize_icd10(parts[0].strip())
        return code
    elif len(parts) == 2:
        start = standardize_icd10(parts[0].strip())
        end = standardize_icd10(parts[1].strip())
        return start, end
    else:
        assert False, f"Unexpected ICD-10 range: {raw_range}"


def process_table_d(icd10_graph, soup):
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
    nodes = [(n, {'type': 'range' if isinstance(n, tuple) else 'code'})
             for n in ({part['target'] for part in parts} |
                       {part['source'] for part in parts})]
    edges = []
    for part in parts:
        edge = (part['source'], part['target'], {'type': 'causes'})
        edges.append(edge)
        if isinstance(part['source'], tuple):
            # Expand range
            codes_in_range = expand_icd10_range(
                icd10_graph, part['source'][0], part['source'][1]
            )
            for code in codes_in_range:
                edges.append((code, part['source'], {'type': 'part_of_range'}))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    acme_file = ICD10_BASE.ensure(url=ACME_URL)
    with open(acme_file, 'r') as fh:
        acme_text = fh.read()
    g = get_icd10_graph()
    soup = BeautifulSoup(acme_text, features='lxml')

    acme_g = process_table_d(g, soup)


