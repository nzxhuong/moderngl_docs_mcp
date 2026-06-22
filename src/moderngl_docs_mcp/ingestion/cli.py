"""CLI entry point for ingesting real dump files into the database.

Usage:
    python -m moderngl_docs_mcp.ingestion.cli moderngl_code_docs.txt code
    python -m moderngl_docs_mcp.ingestion.cli moderngl_info_docs.txt info

Or ingest both in one run:
    python -m moderngl_docs_mcp.ingestion.cli \\
        moderngl_code_docs.txt code \\
        moderngl_info_docs.txt info
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from moderngl_docs_mcp.ingestion.ingest import ingest_dump
from moderngl_docs_mcp.server import EMBEDDING_MODEL_NAME
from moderngl_docs_mcp.storage.db import bootstrap_schema, default_db_path, get_readwrite_connection

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dump_and_type",
        nargs="+",
        metavar="DUMP_FILE DOC_TYPE",
        help="One or more pairs of: a dump file path, then its doc_type label (e.g. code, info).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db_path(),
        help=f"Database path (default: {default_db_path()})",
    )
    args = parser.parse_args()

    if len(args.dump_and_type) % 2 != 0:
        parser.error(
            "Arguments must come in (dump_file, doc_type) pairs, e.g.: "
            "moderngl_code_docs.txt code moderngl_info_docs.txt info"
        )

    pairs = list(zip(args.dump_and_type[0::2], args.dump_and_type[1::2]))

    for dump_path_str, _ in pairs:
        if not Path(dump_path_str).exists():
            logger.error("Dump file not found: %s", dump_path_str)
            sys.exit(1)

    logger.info("Loading embedding model %s...", EMBEDDING_MODEL_NAME)
    logger.info("(first run downloads the model -- this can take a minute or two)")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embedder(text: str) -> list[float]:
        return model.encode(text).tolist()

    db_is_new = not args.db.exists()
    conn = get_readwrite_connection(args.db)
    if db_is_new:
        logger.info("No existing database at %s -- bootstrapping schema.", args.db)
        bootstrap_schema(conn)

    for dump_path_str, doc_type in pairs:
        dump_path = Path(dump_path_str)
        logger.info("Ingesting %s as doc_type=%r...", dump_path, doc_type)
        raw_text = dump_path.read_text(encoding="utf-8")
        stats = ingest_dump(conn, raw_text, doc_type=doc_type, source_file=str(dump_path), embedder=embedder)
        logger.info(
            "  seen=%d added=%d updated=%d skipped=%d",
            stats.sections_seen,
            stats.sections_added,
            stats.sections_updated,
            stats.sections_skipped,
        )

    conn.close()
    logger.info("Done. Database at %s", args.db.resolve())


if __name__ == "__main__":
    main()