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


def load_icd10_definitions(json_file):
    """Load ICD-10 code to definition mappings from JSON file.

    Parameters
    ----------
    json_file : str
        Path to JSON file containing ICD-10 code to definition mappings.

    Returns
    -------
    tuple
        Tuple of (codes, definitions, metadata, hierarchy) first 3 are lists, last one is dict
    """
    logger.debug(f"Loading ICD-10 definitions from {json_file}...")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = []
    definitions = []
    metadata = []
    hierarchy = {}  ## maps higher level codes to their descendants
    for code, entry in sorted(data.items()):
        codes.append(code)
        ## small work around to deal with the presence of lists in the json file in a type safe way.
        definition = entry["definition"]
        if isinstance(definition, list):
            definition = "; ".join(definition)
            hierarchy[code] = entry.get(
                "code"
            )  ## in this case the code attribute stores a list of descendant codes
        assert isinstance(definition, str), "Definition must be parsable as a string"
        definitions.append(definition)
        metadata.append(
            {
                "code": code,
                "name": entry["name"],
                "source": entry["source"],
                "has_definition": entry["has_definition"],
            }
        )

    logger.debug(f"Loaded {len(codes)} ICD-10 codes")

    return codes, definitions, metadata, hierarchy


def generate_embeddings(
    definitions,
    hierarchy: dict[str, list[str]],
    codes: list,
    model_name=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    average_high_order: bool = False,
    normalize=True,
):
    """Generate embeddings for definitions using sentence transformers.

    Parameters
    ----------
    definitions : list of str
        List of text definitions to generate embeddings for.
    hierarchy : dict of list[str]
        Mapping of high level icd-10 code to list of their descendants
    codes : list
        All ICD10 codes used for indexing embeddings
    model_name : str, optional
        Name of the sentence transformer model to use.
        Defaults to 'all-MiniLM-L6-v2'.
    batch_size : int, optional
        Batch size for encoding. Defaults to 32.
    average_high_order: bool, optional
        If to average higher order icd10 codes (if not embeds concatenation of descendants)
    normalize : bool, optional
        Whether to normalize embeddings. Defaults to True.

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

    logger.debug(f"Loading sentence transformer model: {model_name}...")

    model = SentenceTransformer(model_name)

    logger.debug(f"Model loaded (max_seq_length: {model.max_seq_length})")
    logger.debug(f"Generating embeddings (batch_size={batch_size})...")

    embeddings = model.encode(
        definitions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    if average_high_order:
        log_level(f"\nAveraging higher-level ICD10 code embeddings...")
        for high_order_code in hierarchy:
            high_order_idx = codes.index(high_order_code)
            sub_code_idxs = [
                codes.index(sub_code) for sub_code in hierarchy.get(high_order_code, [])
            ]
            if len(sub_code_idxs) < 1:
                raise ValueError(
                    f"ICD10 code {high_order_code} is not properly mapped to leaf nodes"
                )
            embeddings[high_order_idx] = embeddings[sub_code_idxs].mean(axis=0)
            if normalize:
                embeddings[high_order_idx] = embeddings[
                    high_order_idx
                ] / np.linalg.norm(embeddings[high_order_idx])

    logger.debug(f"Generated embeddings shape: {embeddings.shape}")
    logger.debug(f"Embedding dimension: {embeddings.shape[1]}")

    return embeddings


def save_embeddings(
    embeddings,
    output_dir,
):
    """Save embeddings to output directory.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Array of embeddings to save.
    output_dir : str
        Directory path to save embeddings.

    Returns
    -------
    pathlib.Path
        Path to saved embeddings.npy file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.debug(f"Saving embeddings to {output_dir}...")

    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    logger.debug(f"Saved embeddings: {embeddings_file}")

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

    embeddings = np.load(embeddings_file.as_posix())
    with open(definitions_file, "r", encoding="utf-8") as f:
        definitions_data = json.load(f)

    return embeddings, definitions_data


def generate_icd10_embeddings(
    model_name=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    mrconso_path=None,
    mrdef_path=None,
    umls_api_key=None,
    average_high_order: bool = False,
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
    average_high_order : bool, optional
        If to set the embedding of a higher order ICD10 code (chapter, block, etc) as the average of its sub-classes. Otherwise, just concatenates all subclass detentions and embeds them.

    Returns
    -------
    pathlib.Path
        Path to saved embeddings.npy file.
    """
    from .map_definitions import map_icd10_to_definitions

    logger.debug(f"Output directory: {EMBEDDINGS_BASE.base}")

    # Step 1 — Map ICD-10 codes to definitions
    logger.debug("Step 1: Mapping ICD-10 codes to definitions")

    definitions_json = Path(EMBEDDINGS_BASE.base) / "icd10_code_to_definition.json"

    _ = map_icd10_to_definitions(
        mrconso_path=mrconso_path,
        mrdef_path=mrdef_path,
        output_json=definitions_json.as_posix(),
        umls_api_key=umls_api_key,
    )

    # Step 2 — Load definitions
    codes, definitions, _, hierarchy = load_icd10_definitions(
        definitions_json
    )
    # Step 3 — Load embeddings
    embeddings_file = Path(EMBEDDINGS_BASE.base) / "embeddings.npy"

    if embeddings_file.is_file():
        logger.debug("Step 3: Loading existing embeddings")
        logger.debug(f"Embeddings file already exists — loading from {embeddings_file}")

        embeddings = np.load(embeddings_file.as_posix())

        logger.debug(f"Loaded embeddings shape: {embeddings.shape}")
        logger.debug(f"Loaded definitions for {len(codes)} codes")
    else:
        logger.debug("Step 3: Generating embeddings")

        embeddings = generate_embeddings(
            definitions,
            model_name=model_name,
            batch_size=batch_size,
            average_high_order=average_high_order,
            codes=codes,
            hierarchy=hierarchy,
        )
        embeddings_file = save_embeddings(
            embeddings,
            EMBEDDINGS_BASE.base,
        )

    logger.debug("Summary:")
    logger.debug(f"  Total codes processed: {len(codes):,}")
    logger.debug(f"  Embedding dimension: {embeddings.shape[1]}")
    logger.debug(f"  Embeddings shape: {embeddings.shape}")
    logger.debug(f"All files saved to: {EMBEDDINGS_BASE.base}/")
    logger.debug("  - embeddings.npy")
    logger.debug("  - icd10_code_to_definition.json")
    logger.debug("You can reconstruct code index with:")
    logger.debug("  from openacme.generate_embeddings import get_code_index")
    logger.debug("  code_index = get_code_index(definitions_data)")

    return embeddings_file
