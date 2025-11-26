"""Generate embeddings for ICD-10 codes from definitions."""

import json
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

from openacme import OPENACME_BASE

# Pystow module for embeddings
EMBEDDINGS_BASE = OPENACME_BASE.module("icd10_embeddings")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


def load_icd10_definitions(json_file, verbose=True):
    """Load ICD-10 code to definition mappings from JSON file.

    Parameters
    ----------
    json_file : str
        Path to JSON file containing ICD-10 code to definition mappings.
    verbose : bool, optional
        Whether to print progress messages. Defaults to True.

    Returns
    -------
    tuple
        Tuple of (codes, definitions, metadata) lists.
    """
    if verbose:
        print(f"Loading ICD-10 definitions from {json_file}...")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = []
    definitions = []
    metadata = []

    for code, entry in sorted(data.items()):
        codes.append(code)
        definitions.append(entry["definition"])
        metadata.append(
            {
                "code": code,
                "name": entry["name"],
                "source": entry["source"],
                "has_definition": entry["has_definition"],
            }
        )

    if verbose:
        print(f"  ✓ Loaded {len(codes)} ICD-10 codes")

    return codes, definitions, metadata


def generate_embeddings(
    definitions,
    model_name=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    normalize=True,
    verbose=True,
):
    """Generate embeddings for definitions using sentence transformers.

    Parameters
    ----------
    definitions : list of str
        List of text definitions to generate embeddings for.
    model_name : str, optional
        Name of the sentence transformer model to use.
        Defaults to 'all-MiniLM-L6-v2'.
    batch_size : int, optional
        Batch size for encoding. Defaults to 32.
    normalize : bool, optional
        Whether to normalize embeddings. Defaults to True.
    verbose : bool, optional
        Whether to print progress messages. Defaults to True.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_definitions, embedding_dim) containing embeddings.
    """

    if SentenceTransformer is None:  # noqa: SIM108
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )

    if verbose:
        print(f"\nLoading sentence transformer model: {model_name}...")

    model = SentenceTransformer(model_name)

    if verbose:
        print(f"  ✓ Model loaded (max_seq_length: {model.max_seq_length})")
        print(f"\nGenerating embeddings (batch_size={batch_size})...")

    embeddings = model.encode(
        definitions,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    if verbose:
        print(f"  ✓ Generated embeddings shape: {embeddings.shape}")
        print(f"  ✓ Embedding dimension: {embeddings.shape[1]}")

    return embeddings


def save_embeddings(
    codes,
    embeddings,
    metadata,
    output_dir,
    definitions_json=None,
    definitions_csv=None,
    verbose=True,
):
    """Save embeddings and optionally copy definition files.

    Parameters
    ----------
    codes : list of str
        List of ICD-10 codes corresponding to embeddings.
    embeddings : numpy.ndarray
        Array of embeddings to save.
    metadata : list of dict
        List of metadata dictionaries for each code.
    output_dir : str
        Directory path to save embeddings and definition files.
    definitions_json : str, optional
        Path to source JSON definitions file to copy. Defaults to None.
    definitions_csv : str, optional
        Path to source CSV definitions file to copy. Defaults to None.
    verbose : bool, optional
        Whether to print progress messages. Defaults to True.

    Returns
    -------
    pathlib.Path
        Path to saved embeddings.npy file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        print(f"\nSaving embeddings to {output_dir}...")

    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    if verbose:
        print(f"  ✓ Saved embeddings: {embeddings_file}")

    # Copy definitions JSON file
    if definitions_json:
        import shutil

        definitions_json_src = Path(definitions_json)
        definitions_json_dst = output_dir / "icd10_code_to_definition.json"
        if definitions_json_src.exists() and definitions_json_src != definitions_json_dst:
            shutil.copy2(definitions_json_src, definitions_json_dst)
            if verbose:
                print(f"  ✓ Copied definitions JSON: {definitions_json_dst}")
        elif definitions_json_src == definitions_json_dst and verbose:
            print("  ✓ Definitions JSON already in output directory")

    # Copy definitions CSV file
    if definitions_csv:
        import shutil

        definitions_csv_src = Path(definitions_csv)
        definitions_csv_dst = output_dir / "icd10_code_to_definition.csv"
        if definitions_csv_src.exists() and definitions_csv_src != definitions_csv_dst:
            shutil.copy2(definitions_csv_src, definitions_csv_dst)
            if verbose:
                print(f"  ✓ Copied definitions CSV: {definitions_csv_dst}")
        elif definitions_csv_src == definitions_csv_dst and verbose:
            print("  ✓ Definitions CSV already in output directory")

    return embeddings_file


def get_code_index(definitions_data):
    """Generate code index mapping from definitions data.

    Parameters
    ----------
    definitions_data : dict
        Dictionary mapping ICD-10 codes to definition data.

    Returns
    -------
    dict
        Dictionary with 'code_to_idx' and 'idx_to_code' mappings.
    """
    codes = sorted(definitions_data.keys())
    code_to_idx = {code: idx for idx, code in enumerate(codes)}

    return {
        "code_to_idx": code_to_idx,
        "idx_to_code": codes,
    }


def load_embeddings(embeddings_base=None):
    """Load embeddings and definitions from embeddings base directory.

    Parameters
    ----------
    embeddings_base : pystow.Module, optional
        Pystow module for embeddings directory. If None, uses default
        EMBEDDINGS_BASE. Defaults to None.

    Returns
    -------
    tuple
        Tuple of (embeddings, definitions_data) where embeddings is a
        numpy.ndarray and definitions_data is a dict.
    """
    if embeddings_base is None:
        embeddings_base = EMBEDDINGS_BASE

    embeddings_file = Path(embeddings_base.base) / "embeddings.npy"
    definitions_file = Path(embeddings_base.base) / "icd10_code_to_definition.json"

    embeddings = np.load(str(embeddings_file))
    with open(definitions_file, "r", encoding="utf-8") as f:
        definitions_data = json.load(f)

    return embeddings, definitions_data


def generate_icd10_embeddings(
    model_name=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    mrconso_path=None,
    mrdef_path=None,
    umls_api_key=None,
    verbose=True,
):
    """Complete pipeline: Map definitions -> Generate embeddings -> Save.

    Parameters
    ----------
    model_name : str, optional
        Name of the sentence transformer model to use.
        Defaults to 'all-MiniLM-L6-v2'.
    batch_size : int, optional
        Batch size for encoding. Defaults to 32.
    mrconso_path : str, optional
        Path to UMLS MRCONSO.RRF file. If None, downloads automatically.
        Defaults to None.
    mrdef_path : str, optional
        Path to UMLS MRDEF.RRF file. If None, downloads automatically.
        Defaults to None.
    umls_api_key : str, optional
        UMLS API key for downloading data. If None, uses UMLS_API_KEY
        environment variable. Defaults to None.
    verbose : bool, optional
        Whether to print progress messages. Defaults to True.

    Returns
    -------
    pathlib.Path
        Path to saved embeddings.npy file.
    """

    from .map_definitions import map_icd10_to_definitions

    if verbose:
        print("=" * 70)
        print("ICD-10 Code -> Definition -> Embedding Pipeline")
        print("=" * 70)
        print(f"\nOutput directory: {EMBEDDINGS_BASE.base}")

    # Step 1 — Map ICD-10 codes to definitions
    if verbose:
        print("\nStep 1: Mapping ICD-10 codes to definitions")
        print("-" * 70)

    definitions_json = Path(EMBEDDINGS_BASE.base) / "icd10_code_to_definition.json"
    definitions_csv = Path(EMBEDDINGS_BASE.base) / "icd10_code_to_definition.csv"

    icd10_data = map_icd10_to_definitions(
        mrconso_path=mrconso_path,
        mrdef_path=mrdef_path,
        output_json=str(definitions_json),
        output_csv=str(definitions_csv),
        umls_api_key=umls_api_key,
        verbose=verbose,
    )

    # Step 2 — Load definitions and generate embeddings
    if verbose:
        print("\nStep 2: Generating embeddings")
        print("-" * 70)

    codes, definitions, metadata = load_icd10_definitions(definitions_json, verbose=verbose)

    embeddings = generate_embeddings(
        definitions,
        model_name=model_name,
        batch_size=batch_size,
        verbose=verbose,
    )

    embeddings_file = save_embeddings(
        codes,
        embeddings,
        metadata,
        EMBEDDINGS_BASE.base,
        definitions_json=str(definitions_json),
        definitions_csv=str(definitions_csv),
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("✓ Pipeline Complete!")
        print("=" * 70)
        print("\nSummary:")
        print(f"  Total codes processed: {len(codes):,}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"\nAll files saved to: {EMBEDDINGS_BASE.base}/")
        print("  - embeddings.npy")
        print("  - icd10_code_to_definition.json")
        print("  - icd10_code_to_definition.csv")
        print("\nYou can reconstruct code index with:")
        print("  from openacme.generate_embeddings import get_code_index")
        print("  code_index = get_code_index(definitions_data)")

    return embeddings_file
