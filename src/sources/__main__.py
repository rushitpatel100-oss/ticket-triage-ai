"""Download and prepare the real-world datasets.

Run (needs internet, e.g. on Google Colab):
    python -m src.sources                      # all three sources
    python -m src.sources --only techqa itsm_log
    python -m src.sources --skip-download      # reuse files already in data/raw/

Writes data/processed/<source>/ and results/real_data_summary.json.
"""

import argparse
import json
from pathlib import Path

from src import config
from src.sources import common, itsm_log, stackexchange, techqa

SOURCES = ("techqa", "stackexchange", "itsm_log")


def run(args) -> dict:
    summary = {}
    if "techqa" in args.only:
        in_dir = args.techqa_dir or (common.raw_dir(techqa.NAME) if args.skip_download else techqa.download())
        docs, queries = techqa.load(in_dir)
        common.save_tables(techqa.NAME, docs, queries)
        summary["techqa"] = common.summarise_tables(docs, queries)

    if "stackexchange" in args.only:
        in_dir = args.se_dir
        if in_dir is None:
            in_dir = common.raw_dir(stackexchange.NAME)
            if not args.skip_download:
                stackexchange.download(args.se_sites, args.se_max_files, in_dir)
        docs, queries = stackexchange.load(args.se_sites, args.se_max_questions, in_dir=in_dir)
        common.save_tables(stackexchange.NAME, docs, queries)
        summary["stackexchange"] = common.summarise_tables(docs, queries)
        summary["stackexchange"]["sites"] = list(args.se_sites)

    if "itsm_log" in args.only:
        path = args.uci_path or (common.raw_dir(itsm_log.NAME) / itsm_log.ZIP_NAME if args.skip_download
                                 else itsm_log.download())
        inc, ops = itsm_log.load(path)
        out = common.processed_dir(itsm_log.NAME)
        out.mkdir(parents=True, exist_ok=True)
        inc.to_parquet(out / "incidents.parquet", index=False)
        summary["itsm_log"] = ops
    return summary


def write_summary(new: dict) -> dict:
    """Merge into results/real_data_summary.json so running one source keeps the others."""
    path = config.RESULTS_DIR / "real_data_summary.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(new)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))
    return merged


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--skip-download", action="store_true", help="use files already in data/raw/")
    parser.add_argument("--techqa-dir", type=Path)
    parser.add_argument("--se-dir", type=Path)
    parser.add_argument("--uci-path", type=Path)
    parser.add_argument("--se-sites", nargs="+", default=list(config.STACKEXCHANGE_SITES))
    parser.add_argument("--se-max-files", type=int, default=config.STACKEXCHANGE_MAX_FILES)
    parser.add_argument("--se-max-questions", type=int, default=config.STACKEXCHANGE_MAX_QUESTIONS)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    summary = write_summary(run(parse_args(argv)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
