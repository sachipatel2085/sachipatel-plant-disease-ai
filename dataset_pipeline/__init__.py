"""
Crop Health Dataset Pipeline Package.
"""

from .config import (
    DATA_ROOT,
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    MANIFESTS_DIR,
    REJECTS_DIR,
    ensure_directories
)
from .validator import ImageValidator
from .deduplicator import ImageDeduplicator, DeduplicationIndex
from .taxonomy_mapper import TaxonomyMapper
from .sources import RESEARCH_SOURCES, get_source_metadata, list_supported_sources
from .balancer_and_splitter import DatasetBalancerAndSplitter
from .manifest_generator import ManifestGenerator
from .ingestor import DatasetIngestor

__all__ = [
    "DATA_ROOT",
    "RAW_DIR",
    "PROCESSED_DIR",
    "SPLITS_DIR",
    "MANIFESTS_DIR",
    "REJECTS_DIR",
    "ensure_directories",
    "ImageValidator",
    "ImageDeduplicator",
    "DeduplicationIndex",
    "TaxonomyMapper",
    "RESEARCH_SOURCES",
    "get_source_metadata",
    "list_supported_sources",
    "DatasetBalancerAndSplitter",
    "ManifestGenerator",
    "DatasetIngestor"
]
