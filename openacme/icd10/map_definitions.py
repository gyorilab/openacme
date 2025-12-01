"""Map ICD-10 codes to definitions with synonyms."""

import zipfile
from pathlib import Path
from collections import defaultdict
import re
import json
import shutil
import os
import logging
import tqdm

from lxml import etree as ET

from .. import OPENACME_BASE
from . import ICD10_BASE, ICD10_XML_URL

# Pystow modules for organizing data files
UMLS_BASE = OPENACME_BASE.module("umls")
EMBEDDINGS_BASE = OPENACME_BASE.module("icd10_embeddings")

# UMLS download configuration
# Uses UMLS Download API: https://uts-ws.nlm.nih.gov/download
# Requires API key from https://uts.nlm.nih.gov/uts/profile
UMLS_DOWNLOAD_API_BASE = "https://uts-ws.nlm.nih.gov/download"
UMLS_VERSION = "2025AB"
DEFAULT_MRCONSO_FNAME = "MRCONSO.RRF"
DEFAULT_MRDEF_FNAME = "MRDEF.RRF"

# Source priority for definitions
SOURCE_PRIORITY = ["MSH", "CSP", "NCI", "HPO", "SNOMEDCT_US", "MEDLINEPLUS"]

logger = logging.getLogger(__name__)


def is_valid_diagnosis_code(code):
    """Check if code is a valid ICD-10 diagnosis code.

    Parameters
    ----------
    code : str
        ICD-10 code to validate.

    Returns
    -------
    bool
        True if code is a valid diagnosis code, False otherwise.
    """
    if len(code) <= 2 or "-" in code or code.endswith(":"):
        return False
    return bool(re.match(r"^[A-Z]\d{2}(\.\d+)?$", code))


def extract_icd10_codes(icd10_zip_path):
    """Extract ICD-10 codes from XML zip file.

    Parameters
    ----------
    icd10_zip_path : str
        Path to ICD-10 XML zip file.

    Returns
    -------
    dict
        Dictionary mapping ICD-10 codes to their names.
    """
    logger.info("Extracting ICD-10 codes from XML...")

    icd10_codes = {}
    with zipfile.ZipFile(icd10_zip_path, "r") as zf:
        with zf.open("icd102019en.xml") as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for class_elem in root.iter("Class"):
                code = class_elem.get("code")
                if code:
                    name = None
                    for rubric in class_elem.findall(".//Rubric"):
                        label = rubric.find("Label")
                        if label is not None:
                            name = label.text
                            break
                    if name:
                        icd10_codes[code] = name

    valid_codes = {
        code: name
        for code, name in icd10_codes.items()
        if is_valid_diagnosis_code(code)
    }

    logger.info(f"  ✓ Extracted {len(valid_codes)} valid ICD-10 diagnosis codes")

    return valid_codes


def collect_strings_from_mrconso(mrconso_path, valid_codes):
    """Collect all strings/synonyms for ICD-10 codes from MRCONSO.RRF.

    Parameters
    ----------
    mrconso_path : str
        Path to UMLS MRCONSO.RRF file.
    valid_codes : dict
        Dictionary of valid ICD-10 codes to filter by.

    Returns
    -------
    tuple
        Tuple of (code_to_cuis, code_to_strings) dictionaries.
    """
    logger.info("Collecting all strings/synonyms from MRCONSO.RRF...")

    code_to_cuis = defaultdict(set)
    code_to_strings = defaultdict(list)

    with open(mrconso_path, "r", encoding="utf-8", errors="ignore") as f:
        line_count = 0
        for line in tqdm.tqdm(f, desc="Processing MRCONSO.RRF", unit="lines"):
            line_count += 1
            parts = line.strip().split("|")
            if len(parts) >= 15:
                sab = parts[11]
                if sab in ("ICD10", "ICD10CM", "ICD10AM"):
                    cui = parts[0]
                    code = parts[13]
                    string = parts[14]
                    ispref = parts[6] if len(parts) > 6 else ""
                    tty = parts[12] if len(parts) > 12 else ""

                    if code in valid_codes:
                        code_to_cuis[code].add(cui)
                        code_to_strings[code].append(
                            {
                                "string": string,
                                "ispref": ispref == "Y",
                                "tty": tty,
                                "sab": sab,
                            }
                        )

        logger.info(f"  ✓ Found strings for {len(code_to_strings)} codes")
        logger.info(
            "  ✓ Codes with multiple strings: "
            f"{sum(1 for s in code_to_strings.values() if len(s) > 1):,}"
        )

    return code_to_cuis, code_to_strings


def load_definitions_from_mrdef(mrdef_path, all_cuis):
    """Load definitions from MRDEF.RRF for given CUIs.

    Parameters
    ----------
    mrdef_path : str
        Path to UMLS MRDEF.RRF file.
    all_cuis : set
        Set of CUI identifiers to load definitions for.

    Returns
    -------
    dict
        Dictionary mapping CUIs to lists of (source, definition) tuples.
    """
    logger.info("Loading definitions from MRDEF.RRF...")

    cui_to_definitions = defaultdict(list)

    with open(mrdef_path, "r", encoding="utf-8", errors="ignore") as f:
        line_count = 0
        for line in tqdm.tqdm(f, desc="Processing MRDEF.RRF", unit="lines"):
            line_count += 1
            parts = line.strip().split("|")
            if len(parts) >= 6:
                cui = parts[0]
                if cui in all_cuis:
                    sab = parts[4] if len(parts) > 4 else ""
                    definition = parts[5] if len(parts) > 5 else ""
                    if definition and definition.strip():
                        cui_to_definitions[cui].append((sab, definition))

    logger.info(f"  ✓ Found definitions for {len(cui_to_definitions)} CUIs")

    return cui_to_definitions


def get_best_definition(cuis, cui_to_definitions, source_priority=SOURCE_PRIORITY):
    """Get best definition from CUIs based on source priority.

    Parameters
    ----------
    cuis : set
        Set of CUI identifiers to search for definitions.
    cui_to_definitions : dict
        Dictionary mapping CUIs to lists of (source, definition) tuples.
    source_priority : list, optional
        List of source abbreviations in priority order.
        Defaults to SOURCE_PRIORITY.

    Returns
    -------
    tuple
        Tuple of (best_definition, best_source) or (None, None) if not found.
    """
    best_def = None
    best_source = None
    best_priority = float("inf")

    for cui in cuis:
        if cui in cui_to_definitions:
            for sab, definition in cui_to_definitions[cui]:
                priority = len(source_priority)
                if sab in source_priority:
                    priority = source_priority.index(sab)
                if priority < best_priority:
                    best_priority = priority
                    best_def = definition
                    best_source = sab

    return best_def, best_source


def combine_strings_and_definition(strings, definition):
    """Combine synonyms with definition for richer text.

    Parameters
    ----------
    strings : list of dict
        List of string dictionaries with 'string' keys.
    definition : str
        Definition text to combine with synonyms.

    Returns
    -------
    str
        Combined text with synonyms and definition.
    """
    unique_strings = []
    seen = set()

    for s in strings:
        string_lower = s["string"].lower().strip()
        if string_lower not in seen:
            seen.add(string_lower)
            unique_strings.append(s["string"])

    if len(unique_strings) > 1:
        synonyms_text = "; ".join(unique_strings[:5])  # limit to 5 synonyms
        if definition:
            combined = f"{synonyms_text}. {definition}"
        else:
            combined = synonyms_text
    else:
        if definition:
            combined = f"{unique_strings[0]}. {definition}"
        else:
            combined = unique_strings[0] if unique_strings else ""

    return combined


def _get_umls_zip_url():
    """Get download URL for UMLS Metathesaurus Full Subset zip file.

    Returns
    -------
    str
        URL to UMLS Metathesaurus Full Subset zip file.
    """
    return (
        f"https://download.nlm.nih.gov/umls/kss/{UMLS_VERSION}/"
        f"umls-{UMLS_VERSION}-metathesaurus-full.zip"
    )


def _ensure_umls_files(api_key=None):
    """Ensure UMLS files (MRCONSO.RRF, MRDEF.RRF) are available.

    Uses pystow's UMLS_BASE to store data under ~/.data/openacme/umls.
    Downloads the UMLS zip only if it is missing, and extracts
    MRCONSO.RRF and MRDEF.RRF as regular files.

    Parameters
    ----------
    api_key : str, optional
        UMLS API key for downloading data. If None, uses UMLS_API_KEY
        environment variable. Defaults to None.

    Returns
    -------
    tuple
        Tuple of (mrconso_path, mrdef_path) Path objects.

    Raises
    ------
    ValueError
        If API key is not provided and UMLS_API_KEY environment variable
        is not set.
    FileNotFoundError
        If required files are not found in the downloaded zip.
    """
    api_key = api_key or os.getenv("UMLS_API_KEY")
    if not api_key:
        raise ValueError(
            "UMLS API key required. Provide via api_key parameter or set "
            "UMLS_API_KEY environment variable.\n"
            "Get your API key from: https://uts.nlm.nih.gov/uts/profile"
        )

    logger.info("Ensuring UMLS data files are available...")

    # Files should live directly under the UMLS module base directory
    mrconso_path = Path(UMLS_BASE.base) / DEFAULT_MRCONSO_FNAME
    mrdef_path = Path(UMLS_BASE.base) / DEFAULT_MRDEF_FNAME

    # Fast path: if both extracted files already exist, we're done
    if mrconso_path.exists() and mrdef_path.exists():
        logger.info("✓ UMLS files already exist — skipping download and extraction")
        return mrconso_path, mrdef_path

    # Name and expected location of the UMLS master zip
    zip_filename = f"umls-{UMLS_VERSION}-metathesaurus-full.zip"
    zip_path = Path(UMLS_BASE.base) / zip_filename

    # Check if the zip already exists (before calling ensure)
    zip_already_present = zip_path.is_file()

    # Build authenticated URL and ensure the zip is present (downloads if missing)
    zip_url = _get_umls_zip_url()
    authenticated_url = f"{UMLS_DOWNLOAD_API_BASE}?url={zip_url}&apiKey={api_key}"
    zip_path = Path(
        UMLS_BASE.ensure(
            name=zip_filename,
            url=authenticated_url,
        )
    )

    if zip_already_present:
        logger.info(f"  ✓ Using cached zip file: {zip_path}")
    else:
        logger.info(
            "  ✓ Downloaded UMLS Metathesaurus Full Subset zip to: "
            f"{zip_path}"
        )

    # Extract the two RRF files
    # Files are located at: {VERSION}/META/{FILENAME}
    with zipfile.ZipFile(zip_path, "r") as zf:
        mrconso_in_zip = f"{UMLS_VERSION}/META/{DEFAULT_MRCONSO_FNAME}"
        mrdef_in_zip = f"{UMLS_VERSION}/META/{DEFAULT_MRDEF_FNAME}"

        # Extract MRCONSO.RRF
        if mrconso_in_zip not in zf.namelist():
            raise FileNotFoundError(
                f"Could not find {mrconso_in_zip} in zip. "
                f"Available files (first 20): {zf.namelist()[:20]}"
            )
        mrconso_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(mrconso_in_zip) as source, open(mrconso_path, "wb") as target:
            shutil.copyfileobj(source, target)

        # Extract MRDEF.RRF
        if mrdef_in_zip not in zf.namelist():
            raise FileNotFoundError(
                f"Could not find {mrdef_in_zip} in zip. "
                f"Available files (first 20): {zf.namelist()[:20]}"
            )
        mrdef_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(mrdef_in_zip) as source, open(mrdef_path, "wb") as target:
            shutil.copyfileobj(source, target)

    return mrconso_path, mrdef_path


def map_icd10_to_definitions(
    mrconso_path=None,
    mrdef_path=None,
    output_json=None,
    umls_api_key=None,
):
    """Map ICD-10 codes to definitions with synonyms.

    Uses pystow to automatically download ICD-10 XML zip and UMLS data dump
    if needed. UMLS download requires an API key (set via umls_api_key
    parameter or UMLS_API_KEY environment variable).

    Parameters
    ----------
    mrconso_path : str, optional
        Path to UMLS MRCONSO.RRF file. If None, downloads automatically.
        Defaults to None.
    mrdef_path : str, optional
        Path to UMLS MRDEF.RRF file. If None, downloads automatically.
        Defaults to None.
    output_json : str, optional
        Path to output JSON file. If None, uses default location.
        Defaults to None.
    umls_api_key : str, optional
        UMLS API key for downloading data. If None, uses UMLS_API_KEY
        environment variable. Defaults to None.

    Returns
    -------
    dict
        Dictionary mapping ICD-10 codes to definition data.
    """
    # Use pystow to get ICD-10 zip file (downloads if needed)
    icd10_zip_path = ICD10_BASE.ensure(url=ICD10_XML_URL)

    # Use EMBEDDINGS_BASE for output if not specified
    if output_json is None:
        output_json = EMBEDDINGS_BASE.join("icd10_code_to_definition.json")
    output_json = Path(output_json)

    # Short-circuit: if output file already exists, load and return
    if output_json.is_file():
        logger.info("ICD-10 Code to Definition Mapping")
        logger.info("=" * 70)
        logger.info(f"✓ Output file already exists — loading from cache")
        logger.info(f"  JSON: {output_json}")
        with open(output_json, "r", encoding="utf-8") as f:
            icd10_data = json.load(f)
        logger.info(f"  ✓ Loaded {len(icd10_data):,} mappings")
        logger.info("=" * 70)
        return icd10_data

    # Ensure UMLS files are available (downloads and extracts if needed)
    if mrconso_path is None or mrdef_path is None:
        extracted_mrconso, extracted_mrdef = _ensure_umls_files(
            api_key=umls_api_key,
        )
        if mrconso_path is None:
            mrconso_path = extracted_mrconso
        if mrdef_path is None:
            mrdef_path = extracted_mrdef

    mrconso_path = Path(mrconso_path)
    mrdef_path = Path(mrdef_path)

    logger.info("ICD-10 Code to Definition Mapping")
    logger.info("=" * 70)
    logger.info(f"\nUsing ICD-10 zip: {icd10_zip_path}")
    logger.info(f"Using MRCONSO: {mrconso_path}")
    logger.info(f"Using MRDEF: {mrdef_path}")

    # Step 1: Extract ICD-10 codes
    valid_codes = extract_icd10_codes(str(icd10_zip_path))

    # Step 2: Collect strings from MRCONSO
    code_to_cuis, code_to_strings = collect_strings_from_mrconso(
        str(mrconso_path),
        valid_codes,
    )

    # Step 3: Get all CUIs
    all_cuis = set()
    for cuis in code_to_cuis.values():
        all_cuis.update(cuis)

    # Step 4: Load definitions
    cui_to_definitions = load_definitions_from_mrdef(
        str(mrdef_path),
        all_cuis
    )

    # Step 5: Create mappings
    logger.info("\nCreating mappings...")
    logger.info("-" * 70)

    icd10_data = {}

    for code in valid_codes.keys():
        name = valid_codes[code]
        strings = code_to_strings.get(code, [])
        cuis = code_to_cuis.get(code, set())

        # Get definition
        if cuis:
            definition, def_source = get_best_definition(
                cuis,
                cui_to_definitions,
            )
        else:
            definition, def_source = None, None

        # Combine strings and definition
        if strings:
            combined_text = combine_strings_and_definition(strings, definition)
        else:
            combined_text = definition if definition else name

        if not combined_text:
            combined_text = name

        icd10_data[code] = {
            "code": code,
            "name": name,
            "definition": combined_text,
            "source": def_source if definition else "ICD10_XML",
            "has_definition": definition is not None,
            "num_cuis": len(cuis),
            "num_strings": len(strings),
            "synonyms": [s["string"] for s in strings[:10]],
            "original_definition": definition,
        }

    # Statistics
    codes_with_defs = sum(1 for v in icd10_data.values() if v["has_definition"])
    codes_with_synonyms = sum(
        1 for v in icd10_data.values() if v["num_strings"] > 1
    )

    logger.info("\nResults:")
    logger.info("-" * 70)
    logger.info(f"  Total codes: {len(icd10_data):,}")
    logger.info(
        f"  Codes with UMLS definitions: {codes_with_defs:,} "
        f"({codes_with_defs / len(icd10_data) * 100:.1f}%)"
    )
    logger.info(
        "  Codes with multiple strings/synonyms: "
        f"{codes_with_synonyms:,} "
        f"({codes_with_synonyms / len(icd10_data) * 100:.1f}%)"
    )

    avg_length = (
        sum(len(v["definition"]) for v in icd10_data.values())
        / len(icd10_data)
    )
    logger.info(f"  Average definition length: {avg_length:.1f} chars")

    # Save JSON
    logger.info(f"\nSaving data to {output_json}...")
    logger.info("-" * 70)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(icd10_data, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✓ Saved {len(icd10_data):,} mappings")
    logger.info("\n" + "=" * 70)
    logger.info("✓ Mapping complete!")
    logger.info("=" * 70)

    return icd10_data
