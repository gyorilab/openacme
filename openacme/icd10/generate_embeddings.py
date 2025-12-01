"""Generate embeddings for ICD-10 codes from definitions."""

import json
import logging
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .. import OPENACME_BASE

# Pystow module for embeddings
EMBEDDINGS_BASE = OPENACME_BASE.module("icd10_embeddings")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32

logger = logging.getLogger(__name__)


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
    log_level = logger.info if verbose else logger.debug
    log_level(f"Loading ICD-10 definitions from {json_file}...")

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

    log_level(f"  ✓ Loaded {len(codes)} ICD-10 codes")

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
    log_level = logger.info if verbose else logger.debug

    log_level(f"\nLoading sentence transformer model: {model_name}...")

    model = SentenceTransformer(model_name)

    log_level(f"  ✓ Model loaded (max_seq_length: {model.max_seq_length})")
    log_level(f"\nGenerating embeddings (batch_size={batch_size})...")

    embeddings = model.encode(
        definitions,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    log_level(f"  ✓ Generated embeddings shape: {embeddings.shape}")
    log_level(f"  ✓ Embedding dimension: {embeddings.shape[1]}")

    return embeddings


def save_embeddings(
    embeddings,
    output_dir,
    verbose=True,
):
    """Save embeddings to output directory.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Array of embeddings to save.
    output_dir : str
        Directory path to save embeddings.
    verbose : bool, optional
        Whether to print progress messages. Defaults to True.

    Returns
    -------
    pathlib.Path
        Path to saved embeddings.npy file.
    """
    log_level = logger.info if verbose else logger.debug
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    log_level(f"\nSaving embeddings to {output_dir}...")

    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    log_level(f"  ✓ Saved embeddings: {embeddings_file}")

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
    log_level = logger.info if verbose else logger.debug
    from .map_definitions import map_icd10_to_definitions

    log_level("=" * 70)
    log_level("ICD-10 Code -> Definition -> Embedding Pipeline")
    log_level("=" * 70)
    log_level(f"\nOutput directory: {EMBEDDINGS_BASE.base}")

    # Step 1 — Map ICD-10 codes to definitions
    log_level("\nStep 1: Mapping ICD-10 codes to definitions")
    log_level("-" * 70)

    definitions_json = Path(EMBEDDINGS_BASE.base) / "icd10_code_to_definition.json"

    icd10_data = map_icd10_to_definitions(
        mrconso_path=mrconso_path,
        mrdef_path=mrdef_path,
        output_json=str(definitions_json),
        umls_api_key=umls_api_key,
    )

    # Step 2 — Load definitions
    codes, definitions, metadata = load_icd10_definitions(definitions_json, verbose=verbose)

    # Step 3 — Load embeddings
    embeddings_file = Path(EMBEDDINGS_BASE.base) / "embeddings.npy"
    
    if embeddings_file.is_file():
        log_level("\nStep 3: Loading existing embeddings")
        log_level("-" * 70)
        log_level(f"✓ Embeddings file already exists — loading from cache")
        log_level(f"  File: {embeddings_file}")
        
        embeddings = np.load(str(embeddings_file))
        
        log_level(f"  ✓ Loaded embeddings shape: {embeddings.shape}")
        log_level(f"  ✓ Loaded definitions for {len(codes)} codes")
    else:
        log_level("\nStep 3: Generating embeddings")
        log_level("-" * 70)

        embeddings = generate_embeddings(
            definitions,
            model_name=model_name,
            batch_size=batch_size,
            verbose=verbose,
        )

        embeddings_file = save_embeddings(
            embeddings,
            EMBEDDINGS_BASE.base,
            verbose=verbose,
        )

    log_level("\n" + "=" * 70)
    log_level("✓ Pipeline Complete!")
    log_level("=" * 70)
    log_level("\nSummary:")
    log_level(f"  Total codes processed: {len(codes):,}")
    log_level(f"  Embedding dimension: {embeddings.shape[1]}")
    log_level(f"  Embeddings shape: {embeddings.shape}")
    log_level(f"\nAll files saved to: {EMBEDDINGS_BASE.base}/")
    log_level("  - embeddings.npy")
    log_level("  - icd10_code_to_definition.json")
    log_level("\nYou can reconstruct code index with:")
    log_level("  from openacme.generate_embeddings import get_code_index")
    log_level("  code_index = get_code_index(definitions_data)")

    return embeddings_file
