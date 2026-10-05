"""Real ServiceNow incident event log from an IT company (UCI dataset 498, CC BY 4.0).

The log has no ticket text, only process data: priority, assignment groups, reassignments, reopens, SLA,
knowledge-base use and timestamps. We use it for the operational side of the project: how often real
tickets bounce between teams or get reopened, how long they take, and whether using a knowledge article
goes with faster resolution. Those rates feed the business simulation later.

Each incident appears once per update (event), so we collapse the log to one row per incident.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import config
from src.sources.common import raw_dir

NAME = "itsm_log"
ZIP_NAME = "incident_event_log.zip"

BOOL_COLUMNS = ["active", "made_sla", "knowledge", "u_priority_confirmation"]
ORDINAL_COLUMNS = ["impact", "urgency", "priority"]  # values look like "2 - Medium"
DATE_COLUMNS = ["opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at"]
LAST_VALUE_COLUMNS = ["incident_state", "reassignment_count", "reopen_count", "sys_mod_count", "made_sla",
                      "contact_type", "category", "subcategory", "u_symptom", "impact", "urgency", "priority",
                      "knowledge", "u_priority_confirmation", "close_code"]


def download(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or raw_dir(NAME)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ZIP_NAME
    if not target.exists():
        urllib.request.urlretrieve(config.UCI_INCIDENT_LOG_URL, target)
    return target


def read_event_log(path: Path) -> pd.DataFrame:
    """Read the CSV directly or from inside the UCI zip. '?' marks unknown values."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"no CSV inside {path}: {zf.namelist()[:5]}")
            data = zf.read(members[0])
        df = pd.read_csv(io.BytesIO(data), na_values=["?"], low_memory=False)
    else:
        df = pd.read_csv(path, na_values=["?"], low_memory=False)
    return normalise(df)


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return series.map(lambda v: mapping.get(str(v).strip().lower()) if pd.notna(v) else None).astype("boolean")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"closed_code": "close_code"})
    for col in BOOL_COLUMNS:
        if col in df:
            df[col] = _to_bool(df[col])
    for col in ORDINAL_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col].astype("string").str.extract(r"^\s*(\d+)")[0], errors="coerce").astype("Int64")
    for col in DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce", format="mixed")
    for col in ("reassignment_count", "reopen_count", "sys_mod_count"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def incidents(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse the event log to one row per incident, using its latest event for the final state."""
    order = [c for c in ("number", "sys_mod_count", "sys_updated_at") if c in events]
    events = events.sort_values(order, kind="stable")
    grouped = events.groupby("number", sort=True)

    out = grouped[[c for c in LAST_VALUE_COLUMNS if c in events]].last()  # last non-null value per column
    out["n_events"] = grouped.size()
    out["opened_at"] = grouped["opened_at"].min()
    out["resolved_at"] = grouped["resolved_at"].max()
    out["closed_at"] = grouped["closed_at"].max()
    out["first_group"] = grouped["assignment_group"].first()
    out["last_group"] = grouped["assignment_group"].last()
    out["n_groups"] = grouped["assignment_group"].nunique()

    hours = (out["resolved_at"] - out["opened_at"]).dt.total_seconds() / 3600
    out["resolution_hours"] = hours.where(hours >= 0)
    return out.reset_index()


def _median(series: pd.Series):
    series = series.dropna()
    return round(float(series.median()), 1) if len(series) else None


def summarise(inc: pd.DataFrame, n_events: int) -> dict:
    reassigned = inc["reassignment_count"].fillna(0) > 0
    knowledge = inc["knowledge"].fillna(False).astype(bool)
    return {
        "events": int(n_events),
        "incidents": int(len(inc)),
        "median_events_per_incident": _median(inc["n_events"]),
        "priority_counts": {str(k): int(v) for k, v in inc["priority"].value_counts().sort_index().items()},
        "contact_type_counts": {str(k): int(v) for k, v in inc["contact_type"].value_counts().items()},
        "reassigned_share": round(float(reassigned.mean()), 4),
        "mean_reassignments": round(float(inc["reassignment_count"].fillna(0).mean()), 3),
        "more_than_one_group_share": round(float((inc["n_groups"] > 1).mean()), 4),
        "reopened_share": round(float((inc["reopen_count"].fillna(0) > 0).mean()), 4),
        # The UCI description of made_sla is ambiguous, so we report the raw share of True values.
        "made_sla_true_share": round(float(inc["made_sla"].dropna().astype(bool).mean()), 4),
        "knowledge_used_share": round(float(knowledge.mean()), 4),
        "resolved_share": round(float(inc["resolved_at"].notna().mean()), 4),
        "median_resolution_hours": _median(inc["resolution_hours"]),
        "median_resolution_hours_by_priority": {
            str(k): _median(g) for k, g in inc.groupby("priority")["resolution_hours"]},
        "median_resolution_hours_reassigned": {
            "no": _median(inc.loc[~reassigned, "resolution_hours"]),
            "yes": _median(inc.loc[reassigned, "resolution_hours"])},
        "median_resolution_hours_knowledge_used": {
            "no": _median(inc.loc[~knowledge, "resolution_hours"]),
            "yes": _median(inc.loc[knowledge, "resolution_hours"])},
    }


def load(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = path or raw_dir(NAME) / ZIP_NAME
    events = read_event_log(path)
    inc = incidents(events)
    return inc, summarise(inc, len(events))
