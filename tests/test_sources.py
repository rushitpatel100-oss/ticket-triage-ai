"""Offline tests for the real-data loaders, using small files in each dataset's own format."""

import json
import zipfile

import pandas as pd
import pytest

from src import config


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    return tmp_path


# --- TechQA ------------------------------------------------------------------

def make_techqa(folder):
    folder.mkdir(parents=True)
    records = []
    for split, n in (("TRAIN", 20), ("DEV", 10)):
        for i in range(n):
            impossible = i % 4 == 0
            records.append({
                "id": f"{split}_Q{i:03d}",
                "question": f"Error {split}{i} after upgrade?\n\n\nThe   service   fails to start.",
                "answer": "" if impossible else f"Apply fix pack {i}.",
                "is_impossible": impossible,
                "contexts": [{"filename": f"swg{split}{i}.txt", "text": f"TITLE {i}\nFix pack {i} resolves it."}],
            })
    (folder / "train.json").write_text(json.dumps(records))
    with zipfile.ZipFile(folder / "corpus.zip", "w") as zf:
        for i in range(20):
            zf.writestr(f"corpus/swgTRAIN{i}.txt", f"TITLE {i}\nFix pack {i} resolves it.")
        zf.writestr("corpus/swgEXTRA.txt", "Unrelated technote\nAbout something else.")
        zf.writestr("__MACOSX/corpus/._swgEXTRA.txt", "junk")
    return folder


def test_techqa_load(dirs):
    from src.sources import common, techqa

    docs, queries = techqa.load(make_techqa(dirs / "raw" / "techqa"))
    assert len(queries) == 30
    assert queries["answerable"].sum() == 30 - 8  # every 4th question is impossible
    # corpus (21 docs) + the 10 DEV supporting docs that are missing from the corpus
    assert len(docs) == 31 and docs["doc_id"].is_unique
    assert "techqa:swgEXTRA" in set(docs["doc_id"])
    assert docs.loc[docs["doc_id"] == "techqa:swgTRAIN3", "title"].item() == "TITLE 3"
    assert "   " not in queries["text"].iloc[0] and "\n\n\n" not in queries["text"].iloc[0]
    # official DEV questions become the test set; TRAIN is split into train/val
    test_ids = set(queries.loc[queries["split"] == "test", "query_id"])
    assert test_ids == {f"techqa:DEV_Q{i:03d}" for i in range(10)}
    assert set(queries["split"]) == {"train", "val", "test"}
    common.check_tables(docs, queries)


def test_techqa_falls_back_to_stratified_split(dirs):
    from src.sources import techqa

    folder = make_techqa(dirs / "raw" / "techqa")
    records = json.loads((folder / "train.json").read_text())
    for i, rec in enumerate(records):
        rec["id"] = f"Q{i}"
    (folder / "train.json").write_text("\n".join(json.dumps(r) for r in records))  # JSON lines this time
    _, queries = techqa.load(folder)
    assert set(queries["split"]) == {"train", "val", "test"}
    for _, part in queries.groupby("split"):
        assert part["answerable"].nunique() == 2


# --- Stack Exchange ------------------------------------------------------------

def make_stackexchange(folder, site="superuser.com", n=60):
    rows = []
    for q in range(n):
        answers = [
            {"answer_id": 1000 + q, "author": "a", "author_id": 1, "author_profile": "", "pm_score": 3,
             "selected": q % 6 != 0, "text": f"<p>Run <code>sfc /scannow</code> &amp; reboot, case {q}.</p>"},
            {"answer_id": 5000 + q, "author": "b", "author_id": 2, "author_profile": "", "pm_score": 1,
             "selected": False, "text": "<p>Have you tried turning it off and on again?</p>"},
        ]
        rows.append({"qid": q, "question": f"<p>Windows update {q} fails with error 0x8000{q}</p>",
                     "answers": answers, "date": "2020-01-01", "metadata": []})
    out = folder / "data" / site
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(out / "train-00000-of-00001.parquet", index=False)
    return folder


def test_stackexchange_load(dirs):
    from src.sources import common, stackexchange

    folder = make_stackexchange(dirs / "raw" / "stackexchange")
    docs, queries = stackexchange.load(["superuser.com"], max_questions=1000, holdout_share=0.25, in_dir=folder)

    assert len(queries) == 50  # every 6th question has no accepted answer
    assert docs["text"].str.contains("sfc /scannow & reboot").all()  # HTML stripped, entity decoded
    assert not docs["text"].str.contains("turning it off").any()     # only accepted answers
    assert queries["text"].iloc[0].startswith("Windows update")
    assert docs["url"].str.startswith("https://superuser.com/a/").all()

    held_out = queries[~queries["answerable"]]
    assert 0 < len(held_out) < len(queries)
    assert len(docs) == queries["answerable"].sum()
    assert held_out["gold_doc_ids"].map(len).eq(0).all()
    removed = {f"se-superuser:a{1000 + int(q.split(':q')[1])}" for q in held_out["query_id"]}
    assert removed.isdisjoint(set(docs["doc_id"]))  # their answers really are gone from the knowledge base
    common.check_tables(docs, queries)


def test_stackexchange_caps_questions_per_site(dirs):
    from src.sources import stackexchange

    folder = make_stackexchange(dirs / "raw" / "stackexchange")
    _, queries = stackexchange.load(["superuser.com"], max_questions=10, holdout_share=0.0, in_dir=folder)
    assert len(queries) == 10 and queries["answerable"].all()


# --- ServiceNow incident log ---------------------------------------------------

EVENT_COLUMNS = ["number", "incident_state", "active", "reassignment_count", "reopen_count", "sys_mod_count",
                 "made_sla", "caller_id", "opened_by", "opened_at", "sys_created_by", "sys_created_at",
                 "sys_updated_by", "sys_updated_at", "contact_type", "location", "category", "subcategory",
                 "u_symptom", "cmdb_ci", "impact", "urgency", "priority", "assignment_group", "assigned_to",
                 "knowledge", "u_priority_confirmation", "notify", "problem_id", "rfc", "vendor", "caused_by",
                 "closed_code", "resolved_by", "resolved_at", "closed_at"]


def event(number, mod, state, group, reassign, updated, resolved="?", knowledge="false", reopen=0,
          priority="3 - Moderate"):
    row = dict.fromkeys(EVENT_COLUMNS, "?")
    row.update({"number": number, "incident_state": state, "active": "true", "reassignment_count": reassign,
                "reopen_count": reopen, "sys_mod_count": mod, "made_sla": "true", "opened_at": "29/2/2016 01:16",
                "sys_updated_at": updated, "contact_type": "Phone", "category": "Category 55",
                "impact": "2 - Medium", "urgency": "2 - Medium", "priority": priority,
                "assignment_group": group, "knowledge": knowledge, "u_priority_confirmation": "false",
                "resolved_at": resolved, "closed_at": "?"})
    return row


def make_event_log(path):
    rows = [
        # INC1: starts with Group 1, moved to Group 2, resolved 12 hours after opening
        event("INC1", 0, "New", "Group 1", 0, "29/2/2016 01:20"),
        event("INC1", 1, "Active", "Group 2", 1, "29/2/2016 05:00"),
        event("INC1", 2, "Resolved", "Group 2", 1, "29/2/2016 13:16", resolved="29/2/2016 13:16", reopen=1),
        # INC2: one group, a knowledge article was used, resolved after 2 hours
        event("INC2", 0, "New", "?", 0, "29/2/2016 01:30", knowledge="true", priority="2 - High"),
        event("INC2", 1, "Resolved", "Group 3", 0, "29/2/2016 03:16", resolved="29/2/2016 03:16",
              knowledge="true", priority="2 - High"),
    ]
    pd.DataFrame(rows, columns=EVENT_COLUMNS).to_csv(path.with_suffix(".csv"), index=False)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(path.with_suffix(".csv"), "incident_event_log.csv")
    return path


def test_itsm_log_load(dirs):
    from src.sources import itsm_log

    path = make_event_log(dirs / "incident_event_log.zip")
    inc, summary = itsm_log.load(path)
    inc = inc.set_index("number")

    assert inc.loc["INC1", "first_group"] == "Group 1" and inc.loc["INC1", "last_group"] == "Group 2"
    assert inc.loc["INC1", "n_groups"] == 2 and inc.loc["INC1", "n_events"] == 3
    assert inc.loc["INC1", "incident_state"] == "Resolved"
    assert inc.loc["INC1", "resolution_hours"] == pytest.approx(12.0)
    assert inc.loc["INC2", "first_group"] == "Group 3"  # unknown ('?') group is skipped
    assert inc.loc["INC2", "priority"] == 2 and bool(inc.loc["INC2", "knowledge"])
    assert "close_code" in inc.columns

    assert summary["incidents"] == 2 and summary["events"] == 5
    assert summary["reassigned_share"] == 0.5 and summary["reopened_share"] == 0.5
    assert summary["knowledge_used_share"] == 0.5
    assert summary["median_resolution_hours_knowledge_used"] == {"no": 12.0, "yes": 2.0}
    assert summary["priority_counts"] == {"2": 1, "3": 1}


# --- Command line ------------------------------------------------------------

def test_cli_runs_all_sources_offline(dirs):
    from src.sources.__main__ import main

    techqa_dir = make_techqa(dirs / "in" / "techqa")
    se_dir = make_stackexchange(dirs / "in" / "stackexchange")
    uci = make_event_log(dirs / "in" / "log.zip")
    main(["--techqa-dir", str(techqa_dir), "--se-dir", str(se_dir), "--uci-path", str(uci),
          "--se-sites", "superuser.com"])

    summary = json.loads((dirs / "results" / "real_data_summary.json").read_text())
    assert set(summary) == {"techqa", "stackexchange", "itsm_log"}
    assert summary["techqa"]["queries"] == 30
    for name in ("techqa", "stackexchange"):
        assert (dirs / "processed" / name / "docs.parquet").exists()
    assert (dirs / "processed" / "itsm_log" / "incidents.parquet").exists()

    from src.sources.common import load_tables
    docs, queries = load_tables("techqa")
    assert isinstance(queries["gold_doc_ids"].iloc[0], list)

    # Running one source again keeps the others in the summary file
    main(["--only", "techqa", "--techqa-dir", str(techqa_dir)])
    assert set(json.loads((dirs / "results" / "real_data_summary.json").read_text())) == set(summary)
