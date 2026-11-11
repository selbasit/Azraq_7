from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Azraq Calibrated Integrated Fluid Model v3", page_icon="🛢️", layout="wide")

NULLS = (-999.25, -999.0, -9999.0, 999.25)
DATA_DIR = Path(__file__).parent / "data"
SHOW_FILES = {
    "Azraq K-1": DATA_DIR / "Azraq K-1_Shows Summary.csv",
    "Azraq K-2": DATA_DIR / "Azraq K-2_Shows Summary.csv",
    "Azraq K-3": DATA_DIR / "Azraq K-3_Shows Summary.csv",
    "Azraq K-5": DATA_DIR / "Azraq K-5_Shows Summary.csv",
}

DEFAULT_FILES = {
    "Azraq K-1 (SLAM-2)": DATA_DIR / "Azraq K-1_SLAM2.csv",
    "Azraq K-1 (SLAM-3)": DATA_DIR / "Azraq K-1_SLAM3.csv",
    "Azraq K-2 (PEX)": DATA_DIR / "Azraq K-2_PEX.csv",
    "Azraq K-3 (SLAM)": DATA_DIR / "Azraq K-3_SLAM.csv",
    "Azraq K-5 (PEX-HRLA-HNGS-DSI)": DATA_DIR / "Azraq K-5_Merged.las",
}


# Historical K-1 calibration framework extracted from the 2008 well-evaluation package.
# Formation boundaries are mKB / measured-depth intervals as reported in the legacy study.
BUILTIN_FORMATION_TOPS = {
    "Azraq K-1": [
        {"FORMATION": "Amal", "TOP_M": 212.0, "BASE_M": 303.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Baraka", "TOP_M": 303.0, "BASE_M": 343.0, "SALINITY_MIN_PPM": 1300.0, "SALINITY_MAX_PPM": 7500.0},
        {"FORMATION": "Ghazal", "TOP_M": 343.0, "BASE_M": 417.0, "SALINITY_MIN_PPM": 7000.0, "SALINITY_MAX_PPM": 18500.0},
        {"FORMATION": "Zarqa", "TOP_M": 417.0, "BASE_M": 484.0, "SALINITY_MIN_PPM": 8000.0, "SALINITY_MAX_PPM": 22000.0},
        {"FORMATION": "Aradeiba", "TOP_M": 484.0, "BASE_M": 543.0, "SALINITY_MIN_PPM": 12000.0, "SALINITY_MAX_PPM": 30000.0},
        {"FORMATION": "Bentiu", "TOP_M": 543.0, "BASE_M": 838.8, "SALINITY_MIN_PPM": 500.0, "SALINITY_MAX_PPM": 4500.0},
        {"FORMATION": "Abu Gabra", "TOP_M": 838.8, "BASE_M": 2564.0, "SALINITY_MIN_PPM": 3000.0, "SALINITY_MAX_PPM": 75000.0},
    ],
    "Azraq K-5": [
        {"FORMATION": "Nayil", "TOP_M": 144.0, "BASE_M": 267.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Amal", "TOP_M": 267.0, "BASE_M": 460.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Baraka composite", "TOP_M": 460.0, "BASE_M": 606.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Aradeiba", "TOP_M": 606.0, "BASE_M": 684.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Bentiu", "TOP_M": 684.0, "BASE_M": 940.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Abu Gabra", "TOP_M": 940.0, "BASE_M": 1524.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Abu Gabra SQ-1", "TOP_M": 1524.0, "BASE_M": 2007.0, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
        {"FORMATION": "Abu Gabra SQ-2", "TOP_M": 2007.0, "BASE_M": 2617.2, "SALINITY_MIN_PPM": np.nan, "SALINITY_MAX_PPM": np.nan},
    ]
}

# Tested / observed K-1 calibration anchors. These are kept separate from model predictions.
HISTORICAL_TESTS = pd.DataFrame([
    {"WELL":"Azraq K-1","FORMATION":"Amal","TOP_M":213.0,"BASE_M":218.0,"TEST":"DST#5","OBSERVED_FLUID":"Very heavy / waxy oil","BROAD_OBSERVED":"Heavier-oil tendency","API":np.nan,"RATE":"Could not recover","NOTE":"Very heavy, thick and waxy oil."},
    {"WELL":"Azraq K-1","FORMATION":"Zarqa","TOP_M":447.0,"BASE_M":451.0,"TEST":"DST#4a","OBSERVED_FLUID":"Very heavy / waxy oil","BROAD_OBSERVED":"Heavier-oil tendency","API":np.nan,"RATE":"Could not recover","NOTE":"Very heavy, thick and waxy oil."},
    {"WELL":"Azraq K-1","FORMATION":"Bentiu","TOP_M":544.0,"BASE_M":559.0,"TEST":"DST#4","OBSERVED_FLUID":"Very heavy / waxy oil (tar-sand character)","BROAD_OBSERVED":"Heavier-oil tendency","API":np.nan,"RATE":"25.1 bbl formation water + oil","NOTE":"Very heavy, thick and waxy oil; report describes tar-sand character."},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":1670.0,"BASE_M":1680.0,"TEST":"DST#3","OBSERVED_FLUID":"Oil","BROAD_OBSERVED":"Oil-prone","API":35.4,"RATE":"1260 BBL oil","NOTE":"BS&W 0%."},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":2116.0,"BASE_M":2120.0,"TEST":"DST#2","OBSERVED_FLUID":"Oil","BROAD_OBSERVED":"Oil-prone","API":29.86,"RATE":"22.55 BBL oil","NOTE":"BS&W 0%."},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":2190.0,"BASE_M":2195.0,"TEST":"DST#1d","OBSERVED_FLUID":"Slight gas / no influx; very tight","BROAD_OBSERVED":"Gas","API":np.nan,"RATE":"No influx","NOTE":"Petrophysical gas indication but very tight; key producibility warning."},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":2269.0,"BASE_M":2275.0,"TEST":"DST#1c","OBSERVED_FLUID":"Gas","BROAD_OBSERVED":"Gas","API":np.nan,"RATE":"0.165 mscf/d gas","NOTE":"Confirmed gas; useful calibration where mud-gas ratios may appear compositionally rich."},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":2298.0,"BASE_M":2302.0,"TEST":"DST#1a","OBSERVED_FLUID":"Gas","BROAD_OBSERVED":"Gas","API":np.nan,"RATE":"0.058 mscf/d gas","NOTE":"Confirmed gas."},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":2310.0,"BASE_M":2313.0,"TEST":"DST#1","OBSERVED_FLUID":"Condensate + gas","BROAD_OBSERVED":"Condensate / light-oil","API":62.07,"RATE":"180 BBL condensate + 7.7 mscf/d gas","NOTE":"High-API condensate calibration anchor; BS&W 0%."},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":2376.0,"BASE_M":2383.0,"TEST":"DST#1b","OBSERVED_FLUID":"No influx; very tight","BROAD_OBSERVED":"Indeterminate","API":np.nan,"RATE":"No influx","NOTE":"Historical log interpretation water-bearing; very tight test."},
])

PRESSURE_GRADIENTS = pd.DataFrame([
    {"WELL":"Azraq K-1","FORMATION":"Bentiu","TOP_M":544.0,"BASE_M":575.0,"GRADIENT_PSI_FT":0.39,"INTERPRETATION":"Oil-bearing interval"},
    {"WELL":"Azraq K-1","FORMATION":"Bentiu","TOP_M":575.0,"BASE_M":605.0,"GRADIENT_PSI_FT":0.39,"INTERPRETATION":"Oil-bearing interval"},
    {"WELL":"Azraq K-1","FORMATION":"Bentiu","TOP_M":605.0,"BASE_M":838.8,"GRADIENT_PSI_FT":0.43,"INTERPRETATION":"Water character below interpreted POWC; broad historical context"},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":1669.0,"BASE_M":1687.0,"GRADIENT_PSI_FT":0.35,"INTERPRETATION":"Oil-bearing interval"},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":1750.0,"BASE_M":1760.0,"GRADIENT_PSI_FT":0.43,"INTERPRETATION":"Water-bearing interval"},
    {"WELL":"Azraq K-1","FORMATION":"Abu Gabra","TOP_M":1880.0,"BASE_M":1900.0,"GRADIENT_PSI_FT":0.48,"INTERPRETATION":"Water-bearing interval"},
])


K5_FLUID_SAMPLES = pd.DataFrame([
    {"WELL":"Azraq K-5","FORMATION":"Abu Gabra SQ-2","DEPTH_M":2516.5,"TEST":"MDT sample 1","OBSERVED_FLUID":"97% formation water + rare light-brown oil","BROAD_OBSERVED":"Water + trace oil","API":np.nan,"GAS":"No gas","NOTE":"Formation-water dominated sample; oil type not quantified."},
    {"WELL":"Azraq K-5","FORMATION":"Abu Gabra SQ-2","DEPTH_M":2394.0,"TEST":"MDT sample 2","OBSERVED_FLUID":"100% formation water","BROAD_OBSERVED":"Water","API":np.nan,"GAS":"No gas","NOTE":"Water sample; Rw measured at surface conditions in report."},
    {"WELL":"Azraq K-5","FORMATION":"Abu Gabra SQ-2","DEPTH_M":2253.5,"TEST":"MDT sample 3","OBSERVED_FLUID":"95% formation water + trace light-brown oil","BROAD_OBSERVED":"Water + trace oil","API":np.nan,"GAS":"No gas","NOTE":"Formation-water dominated sample; trace oil only."},
])

K5_CORING = pd.DataFrame([
    {"FORMATION":"Abu Gabra SQ-1","TOP_M":1607.5,"BASE_M":1625.5,"RECOVERY_M":15.86,"RECOVERY_PCT":88.11,"NOTE":"~5 m claystone and ~10 m sandstone reported."},
    {"FORMATION":"Abu Gabra SQ-2","TOP_M":2233.0,"BASE_M":2239.5,"RECOVERY_M":6.17,"RECOVERY_PCT":95.0,"NOTE":"Correlation core in SQ-2."},
])

HISTORICAL_K1_CUTOFFS = {"VSH": 0.40, "PHIE": 0.12, "SW": 0.80, "RT": 0.1}

# Canonical curves. Order is preference order when several mnemonics exist.
ALIASES: Dict[str, List[str]] = {
    "DEPTH": ["DEPTH", "DEPT", "MD"],
    "GR": ["GR", "HCGR", "HSGR", "CGR", "SGR"],
    "RT": ["RT", "RD", "RT_HRLT", "RLA5", "RLA4", "RLA3", "RMSL"],
    "RMED": ["RMSL", "RLA3", "RLA2"],
    "RSH": ["RS", "RLA1"],
    "RXO": ["RXOZ", "RLA1"],
    "RHOB": ["ZDEN", "RHOZ"],
    "NPHI": ["CNC", "NPHI"],
    "DT": ["DT", "DTCO", "DTCO3"],
    "PEF": ["PE", "PEFZ"],
    "CALI": ["CAL", "CALI", "HCAL"],
    "BIT": ["BIT", "BS"],
    "SP": ["SPBR", "SP"],
    "TENSION": ["TTEN", "TENS"],
}


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace(list(NULLS), np.nan)
    return out


def _pick(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in df.columns and df[name].notna().any():
            return df[name]
    return pd.Series(np.nan, index=df.index, dtype=float)


def canonicalize(df: pd.DataFrame, source: str, well: str) -> pd.DataFrame:
    raw = _clean_numeric(df)
    out = pd.DataFrame(index=raw.index)
    for canonical, names in ALIASES.items():
        out[canonical] = _pick(raw, names)
    out["SOURCE"] = source
    out["WELL"] = well
    out = out.dropna(subset=["DEPTH"]).sort_values("DEPTH")
    out = out.drop_duplicates(subset=["DEPTH"], keep="last")

    # CNC in SLAM is commonly reported in porosity units (PU); PEX NPHI is fractional.
    if out["NPHI"].dropna().median() > 1.5:
        out["NPHI"] = out["NPHI"] / 100.0
    # Guardrails for clearly invalid values without deleting high-porosity intervals.
    out.loc[~out["NPHI"].between(-0.15, 0.80), "NPHI"] = np.nan
    out.loc[~out["RHOB"].between(1.0, 3.5), "RHOB"] = np.nan
    out.loc[out["RT"] <= 0, "RT"] = np.nan
    out.loc[out["GR"] < 0, "GR"] = np.nan
    return out.reset_index(drop=True)


def infer_well(label: str) -> str:
    text = label.upper().replace("_", " ")
    if "K-1" in text or "K 1" in text:
        return "Azraq K-1"
    if "K-2" in text or "K 2" in text:
        return "Azraq K-2"
    if "K-3" in text or "K 3" in text:
        return "Azraq K-3"
    if "K-5" in text or "K 5" in text:
        return "Azraq K-5"
    return Path(label).stem


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes, source: str) -> pd.DataFrame:
    return canonicalize(pd.read_csv(io.BytesIO(data)), source=source, well=infer_well(source))


@st.cache_data(show_spinner=False)
def read_csv_path(path: str, source: str) -> pd.DataFrame:
    return canonicalize(pd.read_csv(path), source=source, well=infer_well(source))


def _read_las_text(text: str) -> pd.DataFrame:
    """Minimal LAS 2.0 reader for unwrapped, space-delimited files."""
    curves=[]; data=[]; section=None
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('~'):
            section=line.upper()
            continue
        if section and section.startswith('~CURVE'):
            mnem=line.split('.',1)[0].strip().split()[0]
            if mnem: curves.append(mnem)
        elif section and section.startswith('~ASCII'):
            vals=line.split()
            if vals: data.append(vals)
    if not curves or not data:
        raise ValueError('LAS curve or ASCII data section not found')
    n=min(len(curves), min(len(r) for r in data))
    return pd.DataFrame([r[:n] for r in data], columns=curves[:n]).apply(pd.to_numeric, errors='coerce')


@st.cache_data(show_spinner=False)
def read_las_bytes(data: bytes, source: str) -> pd.DataFrame:
    text=data.decode('latin1', errors='ignore')
    return canonicalize(_read_las_text(text), source=source, well=infer_well(source))


@st.cache_data(show_spinner=False)
def read_las_path(path: str, source: str) -> pd.DataFrame:
    return read_las_bytes(Path(path).read_bytes(), source)


def quality_score(row: pd.Series) -> int:
    keys = ["GR", "RT", "RHOB", "NPHI", "DT", "PEF", "CALI", "SP"]
    return int(row[keys].notna().sum())


def merge_k1(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge overlapping K-1 runs using rounded depth and highest curve completeness."""
    if len(parts) == 1:
        return parts[0].copy()
    x = pd.concat(parts, ignore_index=True)
    # Runs are sampled at about 0.076–0.08 m. Round to 0.01 m before selecting best row.
    x["DEPTH_KEY"] = x["DEPTH"].round(2)
    keys = ["GR", "RT", "RHOB", "NPHI", "DT", "PEF", "CALI", "SP"]
    x["QUALITY"] = x[keys].notna().sum(axis=1)
    x = x.sort_values(["DEPTH_KEY", "QUALITY"]).drop_duplicates("DEPTH_KEY", keep="last")
    x["DEPTH"] = x["DEPTH_KEY"]
    x["SOURCE"] = "Merged K-1 runs"
    return x.drop(columns=["DEPTH_KEY", "QUALITY"]).sort_values("DEPTH").reset_index(drop=True)


def load_data(uploaded_files) -> Dict[str, pd.DataFrame]:
    """Load valid depth-log CSVs safely.

    Show-summary or malformed CSV uploads are ignored here rather than allowed to
    produce an empty canonical dataframe and an ``iloc[0]`` crash.  If files
    were uploaded but none is a valid well-log table, the bundled Azraq logs are
    used automatically.
    """
    frames: List[pd.DataFrame] = []

    def _append_if_valid(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        if "DEPTH" not in df.columns or not df["DEPTH"].notna().any():
            return False
        if "WELL" not in df.columns or not df["WELL"].notna().any():
            return False
        frames.append(df)
        return True

    valid_uploads = 0
    if uploaded_files:
        for f in uploaded_files:
            try:
                if str(f.name).lower().endswith('.las'):
                    df = read_las_bytes(f.getvalue(), f.name)
                else:
                    df = read_csv_bytes(f.getvalue(), f.name)
                valid_uploads += int(_append_if_valid(df))
            except Exception:
                # This uploader is for depth-log CSVs.  Semi-structured show
                # summaries and unrelated CSVs are simply ignored.
                continue

    # No uploads, or only non-log/invalid uploads: use the packaged well logs.
    if not uploaded_files or valid_uploads == 0:
        frames = []
        for label, path in DEFAULT_FILES.items():
            if not path.exists():
                continue
            try:
                if str(path).lower().endswith('.las'):
                    _append_if_valid(read_las_path(str(path), label))
                else:
                    _append_if_valid(read_csv_path(str(path), label))
            except Exception:
                continue

    grouped: Dict[str, List[pd.DataFrame]] = {}
    for df in frames:
        if df.empty:
            continue
        well_values = df["WELL"].dropna()
        if well_values.empty:
            continue
        well = str(well_values.iloc[0])
        grouped.setdefault(well, []).append(df)

    result: Dict[str, pd.DataFrame] = {}
    for well, parts in grouped.items():
        valid_parts = [x for x in parts if x is not None and not x.empty]
        if not valid_parts:
            continue
        if well == "Azraq K-1":
            merged = merge_k1(valid_parts)
        else:
            merged = (pd.concat(valid_parts, ignore_index=True)
                        .dropna(subset=["DEPTH"])
                        .sort_values("DEPTH")
                        .drop_duplicates(subset=["DEPTH"], keep="last")
                        .reset_index(drop=True))
        if not merged.empty:
            result[well] = merged
    return result


def normalized_igr(gr: pd.Series, low_q: float, high_q: float) -> Tuple[pd.Series, float, float]:
    valid = gr.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=gr.index), np.nan, np.nan
    gr_clean = float(valid.quantile(low_q))
    gr_shale = float(valid.quantile(high_q))
    if gr_shale <= gr_clean:
        gr_shale = gr_clean + 1.0
    igr = ((gr - gr_clean) / (gr_shale - gr_clean)).clip(0, 1)
    return igr, gr_clean, gr_shale



def parse_formation_tops(uploaded_file=None) -> Dict[str, pd.DataFrame]:
    """Return formation intervals from optional CSV, falling back to built-in K-1 tops.

    Expected optional columns: WELL, FORMATION, TOP_M, BASE_M, and optionally RW_OHM_M,
    SALINITY_MIN_PPM, SALINITY_MAX_PPM.
    """
    result: Dict[str, pd.DataFrame] = {}
    if uploaded_file is not None:
        try:
            df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
            df.columns = [str(c).strip().upper() for c in df.columns]
            required = {"WELL", "FORMATION", "TOP_M", "BASE_M"}
            if required.issubset(df.columns):
                for c in ["TOP_M","BASE_M","RW_OHM_M","SALINITY_MIN_PPM","SALINITY_MAX_PPM"]:
                    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["WELL","FORMATION","TOP_M","BASE_M"])
                for well, g in df.groupby("WELL"):
                    result[str(well)] = g.sort_values("TOP_M").reset_index(drop=True)
        except Exception:
            pass
    for well, rows in BUILTIN_FORMATION_TOPS.items():
        if well not in result:
            result[well] = pd.DataFrame(rows)
    return result


def assign_formations(df: pd.DataFrame, well: str, tops: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    out["FORMATION"] = "Unassigned"
    out["SALINITY_MIN_PPM"] = np.nan
    out["SALINITY_MAX_PPM"] = np.nan
    fdf = tops.get(well, pd.DataFrame())
    if fdf is None or fdf.empty:
        return out
    for _, r in fdf.iterrows():
        mask = out["DEPTH"].between(float(r["TOP_M"]), float(r["BASE_M"]), inclusive="left")
        out.loc[mask, "FORMATION"] = str(r["FORMATION"])
        if "SALINITY_MIN_PPM" in fdf.columns:
            out.loc[mask, "SALINITY_MIN_PPM"] = r.get("SALINITY_MIN_PPM", np.nan)
        if "SALINITY_MAX_PPM" in fdf.columns:
            out.loc[mask, "SALINITY_MAX_PPM"] = r.get("SALINITY_MAX_PPM", np.nan)
    return out


def formation_rw_series(df: pd.DataFrame, global_rw: float, formation_rw: Dict[str, float]) -> pd.Series:
    rw = pd.Series(float(global_rw), index=df.index, dtype=float)
    if "FORMATION" not in df.columns:
        return rw
    for formation, value in formation_rw.items():
        if value is None or not np.isfinite(value) or value <= 0:
            continue
        rw.loc[df["FORMATION"].eq(formation)] = float(value)
    return rw


def calculate_sw_model(out: pd.DataFrame, rw_series: pd.Series, a: float, m: float, n: float,
                       saturation_model: str, rshale: float, rwb: float, shale_bound_phi: float) -> pd.Series:
    """Calculate Sw using selectable screening equations.

    The Dual-Water option is an explicit screening approximation because the legacy
    GEOFRAME DWM parameters (e.g. calibrated bound-water properties) are not available.
    """
    rt = out["RT"].clip(lower=0.01)
    phie = out["PHIE"].clip(lower=0.01)
    phit = out["PHIT"].clip(lower=0.01)
    vsh = out["VSH"].clip(0,1)
    rwv = rw_series.clip(lower=0.001)
    if saturation_model == "Archie":
        return ((a * rwv) / (rt * (phie ** m))).pow(1 / n).clip(0, 1)
    if saturation_model == "Simandoux (screening)":
        # Conductivity = shale conductivity + clean-sand water conductivity.
        water_term = ((1.0/rt) - (vsh/max(rshale,0.01))).clip(lower=0)
        sw_n = water_term * a * rwv / (phie ** m)
        return sw_n.clip(lower=0).pow(1/n).clip(0,1)
    if saturation_model == "Indonesian (screening)":
        lhs = 1.0 / np.sqrt(rt)
        shale_term = (vsh ** (1.0 - vsh/2.0)) / np.sqrt(max(rshale,0.01))
        clean = (lhs - shale_term).clip(lower=0)
        coeff = np.sqrt(a*rwv) / (phie ** (m/2.0))
        return (clean * coeff).clip(lower=0).pow(2.0/n).clip(0,1)
    # Dual-Water screening approximation using total porosity and an estimated bound-water fraction.
    swb = ((vsh * max(shale_bound_phi,0.001)) / phit).clip(0,0.85)
    effective_cw = ((1-swb)/rwv + swb/max(rwb,0.001)).clip(lower=1e-6)
    sw_n = (1.0/rt) * a / ((phit ** m) * effective_cw)
    return sw_n.clip(lower=0).pow(1/n).clip(0,1)


def apply_historical_test_overlay(df: pd.DataFrame, well: str) -> pd.DataFrame:
    out = df.copy()
    out["OBSERVED_TEST_FLUID"] = ""
    out["OBSERVED_TEST"] = ""
    out["OBSERVED_API"] = np.nan
    if well == "Azraq K-5":
        base = out.get("FLUID_CLASS", pd.Series("", index=out.index)).fillna("")
        out["CALIBRATED_FLUID_CLASS"] = base
        for _, t in K5_FLUID_SAMPLES.iterrows():
            mask = (out["DEPTH"] - float(t.DEPTH_M)).abs() <= 0.5
            out.loc[mask, "OBSERVED_TEST_FLUID"] = t.OBSERVED_FLUID
            out.loc[mask, "OBSERVED_TEST"] = t.TEST
            out.loc[mask, "OBSERVED_API"] = t.API
            out.loc[mask, "CALIBRATED_FLUID_CLASS"] = "Confirmed sample: " + str(t.OBSERVED_FLUID)
        return out
    if well != "Azraq K-1":
        out["CALIBRATED_FLUID_CLASS"] = out.get("FLUID_CLASS", "")
        return out
    for _, t in HISTORICAL_TESTS.iterrows():
        mask = out["DEPTH"].between(float(t.TOP_M), float(t.BASE_M), inclusive="both")
        out.loc[mask, "OBSERVED_TEST_FLUID"] = t.OBSERVED_FLUID
        out.loc[mask, "OBSERVED_TEST"] = t.TEST
        out.loc[mask, "OBSERVED_API"] = t.API
    base = out.get("FLUID_CLASS", pd.Series("", index=out.index)).fillna("")
    out["CALIBRATED_FLUID_CLASS"] = base
    obs = out["OBSERVED_TEST_FLUID"].ne("")
    out.loc[obs, "CALIBRATED_FLUID_CLASS"] = "Confirmed: " + out.loc[obs, "OBSERVED_TEST_FLUID"]
    return out


def model_test_validation(df: pd.DataFrame, well: str) -> pd.DataFrame:
    if well != "Azraq K-1": return pd.DataFrame()
    rows=[]
    for _, t in HISTORICAL_TESTS.iterrows():
        g=df[df["DEPTH"].between(float(t.TOP_M),float(t.BASE_M),inclusive="both")]
        fluid = g.get("FLUID_CLASS", pd.Series(dtype=str)).replace("",np.nan).dropna()
        pred = fluid.mode().iloc[0] if not fluid.mode().empty else "Indeterminate"
        pred_broad = pred.replace(" (low-signal)","")
        obs = str(t.BROAD_OBSERVED)
        if obs == "Heavier-oil tendency":
            match = pred_broad in ["Heavier-oil tendency","Oil-prone","Condensate / light-oil"]
        elif obs == "Oil-prone":
            match = pred_broad in ["Oil-prone","Condensate / light-oil","Heavier-oil tendency"]
        elif obs == "Condensate / light-oil":
            match = pred_broad in ["Condensate / light-oil","Oil-prone","Gas"]
        elif obs == "Gas":
            match = pred_broad == "Gas"
        else:
            match = False
        rows.append({
            "Formation":t.FORMATION,"Test":t.TEST,"Top_m":t.TOP_M,"Base_m":t.BASE_M,
            "Observed":t.OBSERVED_FLUID,"API":t.API,"Rate/Result":t.RATE,
            "Model fluid":pred,"Agreement":"Compatible" if match else "Review",
            "Median PHIE":g.PHIE.median() if not g.empty else np.nan,
            "Median Sw":g.SW.median() if not g.empty else np.nan,
            "Pay fraction":g.PAY.mean() if not g.empty else np.nan,
            "Max TG":g.TG.max() if (not g.empty and "TG" in g) else np.nan,
            "Median Wetness":g.WETNESS.median() if (not g.empty and "WETNESS" in g) else np.nan,
            "Note":t.NOTE,
        })
    return pd.DataFrame(rows)

def calculate_petrophysics(
    df: pd.DataFrame,
    well: str,
    tops: Dict[str, pd.DataFrame],
    rw: float,
    formation_rw: Dict[str, float],
    a: float,
    m: float,
    n: float,
    rho_ma: float,
    rho_fl: float,
    shale_model: str,
    saturation_model: str,
    rshale: float,
    rwb: float,
    shale_bound_phi: float,
    gr_low_q: float,
    gr_high_q: float,
    vsh_cut: float,
    phie_cut: float,
    sw_cut: float,
    rt_cut: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = assign_formations(df.copy(), well, tops)
    igr, gr_clean, gr_shale = normalized_igr(out["GR"], gr_low_q, gr_high_q)
    out["IGR"] = igr
    if shale_model == "Larionov (Tertiary)":
        out["VSH"] = (0.083 * (2 ** (3.7 * igr) - 1)).clip(0, 1)
    elif shale_model == "Larionov (Older rocks)":
        out["VSH"] = (0.33 * (2 ** (2 * igr) - 1)).clip(0, 1)
    else:
        out["VSH"] = igr.clip(0, 1)

    denom = rho_ma - rho_fl
    out["PHID"] = ((rho_ma - out["RHOB"]) / denom).clip(-0.10, 0.60) if denom else np.nan
    out["PHIT"] = pd.concat([out["PHID"], out["NPHI"]], axis=1).mean(axis=1, skipna=True).clip(0, 0.60)
    out["PHIE"] = (out["PHIT"] * (1 - out["VSH"])).clip(0, 0.50)

    out["RW_USED"] = formation_rw_series(out, rw, formation_rw)
    out["SATURATION_MODEL"] = saturation_model
    out["SW"] = calculate_sw_model(out, out["RW_USED"], a, m, n, saturation_model, rshale, rwb, shale_bound_phi)
    out["SH"] = 1 - out["SW"]
    out["BVW"] = out["PHIE"] * out["SW"]

    out["RESERVOIR"] = (out["VSH"] <= vsh_cut) & (out["PHIE"] >= phie_cut)
    out["PAY"] = out["RESERVOIR"] & (out["SW"] <= sw_cut) & (out["RT"] >= rt_cut)

    components = pd.DataFrame({
        "clean": (1 - out["VSH"]).clip(0, 1),
        "por": (out["PHIE"] / max(phie_cut * 2.0, 0.20)).clip(0, 1),
        "hc": out["SH"].clip(0, 1),
        "res": (np.log10(out["RT"].clip(lower=0.1)) / 2.0).clip(0, 1),
    })
    out["HC_SCORE"] = components.mean(axis=1, skipna=True)
    out["FLUID_SCREEN"] = np.select(
        [out["PAY"] & (out["HC_SCORE"] >= 0.70), out["PAY"] & (out["HC_SCORE"] >= 0.52)],
        ["Strong HC response", "Moderate HC response"],
        default="Non-pay / uncertain",
    )

    params = {"GR_clean": gr_clean, "GR_shale": gr_shale}
    return out, params

def thickness_summary(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values("DEPTH").copy()
    dz = x["DEPTH"].diff().abs()
    nominal = float(dz[dz > 0].median()) if (dz > 0).any() else 0.0
    # Cap gaps to avoid counting missing depth as rock thickness.
    x["DZ"] = dz.fillna(nominal).clip(upper=max(nominal * 2.5, nominal))
    gross = float(x["DZ"].sum())
    reservoir = float(x.loc[x["RESERVOIR"], "DZ"].sum())
    pay = float(x.loc[x["PAY"], "DZ"].sum())
    return pd.DataFrame({
        "Gross logged (m)": [gross],
        "Net reservoir (m)": [reservoir],
        "Net pay (m)": [pay],
        "N/G reservoir": [reservoir / gross if gross else np.nan],
        "N/G pay": [pay / gross if gross else np.nan],
        "Mean PHIE in pay": [x.loc[x["PAY"], "PHIE"].mean()],
        "Mean Sw in pay": [x.loc[x["PAY"], "SW"].mean()],
        "Mean Rt in pay": [x.loc[x["PAY"], "RT"].mean()],
    })


def contiguous_zones(df: pd.DataFrame, mask_col: str = "PAY", max_gap_factor: float = 2.5) -> pd.DataFrame:
    x = df.sort_values("DEPTH").copy()
    x = x[x[mask_col].fillna(False)].copy()
    if x.empty:
        return pd.DataFrame(columns=["Zone", "Top_m", "Base_m", "Gross_m", "Mean_PHIE", "Mean_SW", "Mean_RT", "HC_Response"])
    step = float(df["DEPTH"].diff().abs().median())
    new_zone = x["DEPTH"].diff().abs().fillna(0) > max(step * max_gap_factor, 0.25)
    x["ZONE_ID"] = new_zone.cumsum() + 1
    rows = []
    for zid, g in x.groupby("ZONE_ID"):
        top, base = float(g["DEPTH"].min()), float(g["DEPTH"].max())
        rows.append({
            "Zone": int(zid), "Top_m": top, "Base_m": base,
            "Gross_m": base - top + step,
            "Mean_PHIE": g["PHIE"].mean(), "Mean_SW": g["SW"].mean(),
            "Mean_RT": g["RT"].mean(),
            "HC_Response": g["FLUID_SCREEN"].mode().iloc[0] if not g["FLUID_SCREEN"].mode().empty else "Uncertain",
        })
    return pd.DataFrame(rows).sort_values("Gross_m", ascending=False).reset_index(drop=True)



def parse_show_summary(path_or_buffer, well: str) -> pd.DataFrame:
    """Parse the semi-structured Azraq mud-log Shows Summary sheets."""
    raw = pd.read_csv(path_or_buffer, header=None)
    rows = []
    for _, r in raw.iterrows():
        # A valid show row needs columns 0..14.  Header/footer/short rows are ignored.
        if len(r) < 15:
            continue
        try:
            no = float(str(r.iloc[0]).strip())
            top = float(str(r.iloc[1]).strip())
            base = float(str(r.iloc[2]).strip())
        except Exception:
            continue
        if not (np.isfinite(no) and np.isfinite(top) and np.isfinite(base)):
            continue
        thickness = pd.to_numeric(pd.Series([r.iloc[3]]), errors="coerce").iloc[0]
        row = {
            "WELL": well, "SHOW_NO": int(no), "FROM_M": top, "TO_M": base,
            "THICKNESS_M": float(thickness) if pd.notna(thickness) else max(base-top, 0),
            "DIRECT_FLUOR": str(r.iloc[4]).strip() if pd.notna(r.iloc[4]) else "",
            "DISTRIBUTION": str(r.iloc[5]).strip() if pd.notna(r.iloc[5]) else "",
            "CUT_FLUOR": str(r.iloc[6]).strip() if pd.notna(r.iloc[6]) else "",
        }
        for name, j in zip(["TG","C1","C2","C3","IC4","NC4","IC5","NC5"], range(7,15)):
            v = pd.to_numeric(pd.Series([r.iloc[j]]), errors="coerce").iloc[0]
            row[name] = float(v) if pd.notna(v) and v >= 0 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["C4"] = out["IC4"] + out["NC4"]
    out["C5"] = out["IC5"] + out["NC5"]
    out["C1_C5"] = out[["C1","C2","C3","C4","C5"]].sum(axis=1)
    heavy = out[["C2","C3","C4","C5"]].sum(axis=1)
    c3plus = out[["C3","C4","C5"]].sum(axis=1)
    out["WETNESS"] = np.where(out["C1_C5"] > 0, 100.0 * heavy / out["C1_C5"], np.nan)
    out["BALANCE"] = np.where(c3plus > 0, (out["C1"] + out["C2"]) / c3plus, np.nan)
    out["CHARACTER"] = np.where(out["C3"] > 0, (out["C4"] + out["C5"]) / out["C3"], np.nan)
    out["C1_C2"] = np.where(out["C2"] > 0, out["C1"] / out["C2"], np.nan)
    out["C1_C3"] = np.where(out["C3"] > 0, out["C1"] / out["C3"], np.nan)
    out["C3PLUS_PCT"] = np.where(out["C1_C5"] > 0, 100.0*c3plus/out["C1_C5"], np.nan)
    out["MID_DEPTH_M"] = (out["FROM_M"] + out["TO_M"]) / 2.0
    return classify_show_fluids(out)


def _haworth_class(wh, bh, ch):
    if not np.isfinite(wh): return "Indeterminate"
    if np.isfinite(bh) and bh > 100 and wh < 0.5: return "Dry gas"
    if wh < 0.5: return "Dry gas / background"
    if wh < 17.5:
        if np.isfinite(bh) and bh < wh:
            return "Condensate / high-GOR oil" if (not np.isfinite(ch) or ch >= 0.5) else "Very wet gas"
        return "Wet gas"
    if wh <= 40:
        return "Oil-prone" if (not np.isfinite(bh) or bh < wh) else "Condensate / volatile oil"
    return "Heavier-oil / residual tendency"


def _pixler_class(r12):
    if not np.isfinite(r12): return "Indeterminate"
    if r12 < 2: return "Residual / very heavy tendency"
    if r12 < 4: return "Low-gravity oil tendency"
    if r12 < 8: return "Medium-oil tendency"
    if r12 < 15: return "Light-oil / condensate tendency"
    if r12 <= 65: return "Gas / condensate tendency"
    return "Very dry gas tendency"


def _broad_fluid(label: str) -> str:
    x = label.lower()
    if "heavy" in x or "residual" in x or "low-gravity" in x: return "Heavier-oil tendency"
    if "oil" in x and "condensate" not in x: return "Oil-prone"
    if "condensate" in x or "volatile" in x or "high-gor" in x: return "Condensate / light-oil"
    if "gas" in x: return "Gas"
    return "Indeterminate"


def classify_show_fluids(df: pd.DataFrame, tg_floor: float = 1000.0) -> pd.DataFrame:
    out = df.copy()
    out["HAWORTH_CLASS"] = [_haworth_class(w,b,c) for w,b,c in zip(out.WETNESS, out.BALANCE, out.CHARACTER)]
    out["PIXLER_CLASS"] = [_pixler_class(v) for v in out.C1_C2]
    haw = out["HAWORTH_CLASS"].map(_broad_fluid)
    pix = out["PIXLER_CLASS"].map(_broad_fluid)
    # Haworth is the primary class; Pixler and fluorescence affect confidence.
    out["FLUID_CLASS"] = haw
    # Low gas responses are retained but explicitly downgraded.
    out.loc[out["TG"] < tg_floor, "FLUID_CLASS"] = out.loc[out["TG"] < tg_floor, "FLUID_CLASS"] + " (low-signal)"
    fluor_text = (out["DIRECT_FLUOR"] + " " + out["CUT_FLUOR"]).str.lower()
    fluor_positive = ~fluor_text.str.contains(r"\bno\b|none", regex=True)
    # Signal confidence: 0 at <=100 ppm, ~1 above 10,000 ppm.
    sig = ((np.log10(out["TG"].clip(lower=100)) - 2.0) / 2.0).clip(0,1)
    agreement = (haw == pix).astype(float)
    ratio_complete = out[["WETNESS","BALANCE","CHARACTER","C1_C2"]].notna().mean(axis=1)
    conf = 0.45*sig + 0.25*agreement + 0.20*ratio_complete + 0.10*fluor_positive.astype(float)
    out["CONFIDENCE_SCORE"] = (100*conf).clip(0,100)
    out["CONFIDENCE"] = pd.cut(out["CONFIDENCE_SCORE"], bins=[-1,45,70,101], labels=["Low","Moderate","High"], right=False).astype(str)
    out["MODEL_NOTE"] = np.where(out["FLUID_CLASS"].str.contains("Heavier", case=False),
        "C1-C5 heavier-oil tendency only; confirm API gravity/viscosity/PVT.",
        "Mud-gas fluid screen; calibrate to tests/samples and drilling conditions.")
    return out


def load_show_data(uploaded_show_files=None) -> Dict[str, pd.DataFrame]:
    """Load mud-log Shows Summary files from uploads first, then bundled data.

    The uploaded file name is used to infer Azraq K-1/K-2/K-3/K-5. If no valid
    uploaded summary exists for a well, the packaged data/ file is used as a
    fallback. This makes the app robust when only app.py is deployed.
    """
    result: Dict[str, pd.DataFrame] = {}

    if uploaded_show_files:
        for f in uploaded_show_files:
            try:
                well = infer_well(f.name)
                if well not in SHOW_FILES:
                    continue
                parsed = parse_show_summary(io.BytesIO(f.getvalue()), well)
                if parsed is not None and not parsed.empty:
                    result[well] = parsed
            except Exception:
                continue

    for well, path in SHOW_FILES.items():
        if well in result:
            continue
        if path.exists():
            try:
                parsed = parse_show_summary(path, well)
                if parsed is not None and not parsed.empty:
                    result[well] = parsed
            except Exception:
                continue
    return result


def integrate_show_intervals(log_df: pd.DataFrame, shows: pd.DataFrame) -> pd.DataFrame:
    out = log_df.copy()
    gas_cols = ["SHOW_NO","TG","C1","C2","C3","C4","C5","WETNESS","BALANCE","CHARACTER","C1_C2","C1_C3","C3PLUS_PCT","FLUID_CLASS","CONFIDENCE_SCORE","CONFIDENCE","DIRECT_FLUOR","CUT_FLUOR"]
    for c in gas_cols:
        out[c] = np.nan if c not in ["FLUID_CLASS","CONFIDENCE","DIRECT_FLUOR","CUT_FLUOR"] else ""
    if shows is None or shows.empty:
        return out
    # Assign show attributes only inside recorded show intervals. In overlaps, retain higher TG.
    best_tg = pd.Series(-np.inf, index=out.index)
    for _, sh in shows.sort_values("TG").iterrows():
        mask = out["DEPTH"].between(sh["FROM_M"], sh["TO_M"], inclusive="both") & (sh["TG"] >= best_tg)
        if not mask.any(): continue
        best_tg.loc[mask] = sh["TG"]
        for c in gas_cols:
            out.loc[mask, c] = sh.get(c, np.nan)
    return out


def gas_log_panel(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=8, shared_yaxes=True, horizontal_spacing=0.018,
        subplot_titles=("GR/Vsh","Rt","PHIE","Sw","TG","C1-C5 wetness","W-B","Fluid class"))
    d=df["DEPTH"]
    fig.add_trace(go.Scatter(x=df.GR,y=d,name="GR",line=dict(width=1)),1,1)
    fig.add_trace(go.Scatter(x=df.VSH*150,y=d,name="Vsh×150",line=dict(width=1)),1,1)
    fig.add_trace(go.Scatter(x=df.RT,y=d,name="Rt",line=dict(width=1)),1,2)
    fig.add_trace(go.Scatter(x=df.PHIE,y=d,name="PHIE",line=dict(width=1)),1,3)
    fig.add_trace(go.Scatter(x=df.SW,y=d,name="Sw",line=dict(width=1)),1,4)
    fig.add_trace(go.Scatter(x=df.TG,y=d,name="TG ppm",line=dict(width=1)),1,5)
    fig.add_trace(go.Scatter(x=df.WETNESS,y=d,name="Wetness %",line=dict(width=1)),1,6)
    fig.add_trace(go.Scatter(x=df.BALANCE,y=d,name="Balance",line=dict(width=1)),1,7)
    class_map={"Gas":1,"Condensate / light-oil":2,"Oil-prone":3,"Heavier-oil tendency":4,"Indeterminate":0}
    broad=df["FLUID_CLASS"].fillna("").str.replace(" (low-signal)","",regex=False)
    cx=broad.map(class_map).fillna(0)
    fig.add_trace(go.Scatter(x=cx,y=d,mode="markers",marker=dict(size=4),name="Fluid screen",text=df["FLUID_CLASS"]),1,8)
    fig.update_yaxes(autorange="reversed",title_text="Depth (m)",row=1,col=1)
    fig.update_xaxes(range=[0,150],row=1,col=1)
    fig.update_xaxes(type="log",range=[-1,3],row=1,col=2)
    fig.update_xaxes(range=[0,0.45],row=1,col=3)
    fig.update_xaxes(range=[0,1],row=1,col=4)
    fig.update_xaxes(type="log",row=1,col=5)
    fig.update_xaxes(range=[0,50],row=1,col=6)
    fig.update_xaxes(type="log",row=1,col=7)
    fig.update_xaxes(tickvals=[0,1,2,3,4],ticktext=["—","Gas","Cond.","Oil","Heavy*"],row=1,col=8)
    fig.update_layout(height=900,title=title,legend=dict(orientation="h",y=1.06),margin=dict(l=20,r=20,t=90,b=20))
    return fig

def ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    candidates = ["GR", "RT", "RHOB", "NPHI", "DT", "PEF", "VSH", "PHIE", "SW"]
    cols = [c for c in candidates if df[c].notna().sum() >= 50]
    return df[cols].replace([np.inf, -np.inf], np.nan), cols


def apply_unsupervised(df: pd.DataFrame, clusters: int, contamination: float) -> pd.DataFrame:
    out = df.copy()
    X, cols = ml_features(out)
    out["FACIES_CLUSTER"] = np.nan
    out["ANOMALY"] = False
    if len(cols) < 2 or len(X) < 100:
        return out
    valid = X.notna().sum(axis=1) >= max(2, math.ceil(len(cols) * 0.5))
    if valid.sum() < 100:
        return out
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    Z = pipe.fit_transform(X.loc[valid])
    k = min(clusters, max(2, valid.sum() // 40))
    out.loc[valid, "FACIES_CLUSTER"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    detector = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
    out.loc[valid, "ANOMALY"] = detector.fit_predict(Z) == -1
    return out


def log_panel(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=6, shared_yaxes=True, horizontal_spacing=0.025,
                        subplot_titles=("GR / Vsh", "Resistivity", "Density–Neutron", "Porosity", "Saturation", "Pay"))
    d = df["DEPTH"]
    fig.add_trace(go.Scatter(x=df["GR"], y=d, name="GR", line=dict(color="green", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df["VSH"] * 150, y=d, name="Vsh×150", line=dict(color="olive", width=1)), 1, 1)
    for c, color in [("RT", "red"), ("RMED", "orange"), ("RSH", "black")]:
        fig.add_trace(go.Scatter(x=df[c], y=d, name=c, line=dict(color=color, width=1)), 1, 2)
    fig.add_trace(go.Scatter(x=df["RHOB"], y=d, name="RHOB", line=dict(color="blue", width=1)), 1, 3)
    fig.add_trace(go.Scatter(x=df["NPHI"], y=d, name="NPHI", line=dict(color="purple", width=1)), 1, 3)
    fig.add_trace(go.Scatter(x=df["PHIT"], y=d, name="PHIT", line=dict(color="gray", width=1)), 1, 4)
    fig.add_trace(go.Scatter(x=df["PHIE"], y=d, name="PHIE", line=dict(color="teal", width=1.4)), 1, 4)
    fig.add_trace(go.Scatter(x=df["SW"], y=d, name="Sw", line=dict(color="blue", width=1)), 1, 5)
    fig.add_trace(go.Scatter(x=df["SH"], y=d, name="Sh", line=dict(color="red", width=1)), 1, 5)
    pay_x = df["PAY"].astype(int)
    fig.add_trace(go.Scatter(x=pay_x, y=d, mode="lines", fill="tozerox", name="Pay", line=dict(color="goldenrod", width=0.5)), 1, 6)
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
    fig.update_xaxes(range=[0, 150], row=1, col=1)
    fig.update_xaxes(type="log", range=[-1, 3], row=1, col=2)
    fig.update_xaxes(range=[1.7, 3.0], row=1, col=3)
    fig.update_xaxes(range=[0, 0.45], row=1, col=4)
    fig.update_xaxes(range=[0, 1], row=1, col=5)
    fig.update_xaxes(range=[0, 1], row=1, col=6)
    fig.update_layout(height=900, title=title, legend=dict(orientation="h", y=1.06), margin=dict(l=20, r=20, t=90, b=20))
    return fig


def correlation_panel(models: Dict[str, pd.DataFrame], curve: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=list(models.keys()))
    for i, (well, df) in enumerate(models.items(), start=1):
        fig.add_trace(go.Scatter(x=df[curve], y=df["DEPTH"], mode="lines", name=well, showlegend=False), 1, i)
        pay = df[df["PAY"]]
        if not pay.empty:
            fig.add_trace(go.Scatter(x=pay[curve], y=pay["DEPTH"], mode="markers", marker=dict(size=3), name=f"{well} pay", showlegend=False), 1, i)
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    fig.update_layout(height=850, title=f"Azraq multi-well correlation view — {curve}", margin=dict(l=20, r=20, t=70, b=20))
    return fig


st.title("🛢️ Azraq Area — Calibrated Integrated Petrophysical + Fluid Model v3.1")
st.caption("Four-well model: K-1, K-2, K-3 and K-5. K-1 uses historical DST/API/pressure calibration; K-5 adds PEX/HRLA/HNGS/DSI logs, formation tops, cores and MDT/CST fluid evidence. Measured/test evidence takes interpretive priority over screening labels.")

with st.sidebar:
    st.header("Data")
    uploads = st.file_uploader(
        "Optional replacement well-log CSV/LAS files", type=["csv","las"], accept_multiple_files=True, key="log_uploads",
        help="Upload depth-based K-1/K-2/K-3/K-5 log CSV or LAS files here.",
    )
    show_uploads = st.file_uploader(
        "Optional Shows Summary CSV files", type=["csv"], accept_multiple_files=True, key="show_uploads",
        help="Upload the matching K-1/K-2/K-3/K-5 Shows Summary files. A K-5 file is used automatically when bundled.",
    )
    tops_upload = st.file_uploader(
        "Optional formation-tops CSV", type=["csv"], accept_multiple_files=False, key="tops_upload",
        help="Columns: WELL, FORMATION, TOP_M, BASE_M; optional RW_OHM_M and salinity columns. Built-in K-1 and K-5 tops are used when no file is supplied.",
    )

    st.header("Interpretation preset")
    preset = st.selectbox("Preset", ["Integrated screening", "Historical K-1 calibration"])
    hist = preset == "Historical K-1 calibration"

    st.header("Petrophysical assumptions")
    rw = st.number_input("Global Rw (ohm·m)", 0.005, 1.0, 0.08, 0.005, format="%.3f")
    a = st.number_input("Archie a", 0.5, 2.0, 1.0, 0.05)
    m = st.number_input("Cementation exponent m", 1.2, 3.5, 2.0, 0.05)
    n = st.number_input("Saturation exponent n", 1.2, 3.5, 2.0, 0.05)
    rho_ma = st.number_input("Matrix density (g/cc)", 2.40, 2.90, 2.65, 0.01)
    rho_fl = st.number_input("Fluid density (g/cc)", 0.80, 1.20, 1.00, 0.01)
    shale_model = st.selectbox("Vsh model", ["Larionov (Tertiary)", "Larionov (Older rocks)", "Linear"])
    saturation_options = ["Archie", "Simandoux (screening)", "Indonesian (screening)", "Dual Water (screening approximation)"]
    saturation_model = st.selectbox("Saturation model", saturation_options, index=3 if hist else 0)
    rshale = st.number_input("Rshale (ohm·m)", 0.1, 50.0, 2.0, 0.1, help="Used by Simandoux/Indonesian screening models.")
    rwb = st.number_input("Bound-water Rw (ohm·m)", 0.005, 1.0, 0.03, 0.005, format="%.3f", help="Used only by the Dual-Water screening approximation.")
    shale_bound_phi = st.number_input("Shale bound-water porosity", 0.01, 0.40, 0.10, 0.01, help="Screening proxy used only by the Dual-Water approximation.")
    gr_low_q = st.slider("Clean GR quantile", 0.01, 0.30, 0.05, 0.01)
    gr_high_q = st.slider("Shale GR quantile", 0.70, 0.99, 0.95, 0.01)

    st.header("Formation-specific Rw")
    use_form_rw = st.checkbox("Use K-1 formation-specific Rw", value=hist, help="Values are user calibration inputs. Historical salinity ranges are shown as context but are not converted automatically into Rw.")
    formation_rw = {}
    with st.expander("K-1 Rw by formation", expanded=False):
        for fm in ["Amal","Baraka","Ghazal","Zarqa","Aradeiba","Bentiu","Abu Gabra"]:
            formation_rw[fm] = st.number_input(f"{fm} Rw", 0.005, 1.0, float(rw), 0.005, key=f"rw_{fm}", format="%.3f")
    if not use_form_rw:
        formation_rw = {}

    st.header("Cutoffs")
    vsh_cut = st.slider("Vsh maximum", 0.10, 0.70, 0.40, 0.01)
    phie_cut = st.slider("PHIE minimum", 0.02, 0.25, 0.12 if hist else 0.08, 0.01)
    sw_cut = st.slider("Sw maximum", 0.20, 0.95, 0.80 if hist else 0.60, 0.01)
    rt_cut = st.number_input("Rt minimum (ohm·m)", 0.1, 100.0, 0.1 if hist else 10.0, 0.5)

    st.header("ML screening")
    clusters = st.slider("Facies clusters", 2, 8, 4)
    contamination = st.slider("Anomaly fraction", 0.01, 0.20, 0.05, 0.01)

try:
    wells = load_data(uploads)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if not wells:
    st.error("No valid Azraq CSV/LAS files were found.")
    st.stop()

formation_tops = parse_formation_tops(tops_upload)

models: Dict[str, pd.DataFrame] = {}
calibration: Dict[str, Dict[str, float]] = {}
shows_by_well = load_show_data(show_uploads)
for well, df in wells.items():
    interpreted, params = calculate_petrophysics(
        df, well, formation_tops, rw, formation_rw if well == "Azraq K-1" else {},
        a, m, n, rho_ma, rho_fl, shale_model, saturation_model, rshale, rwb, shale_bound_phi,
        gr_low_q, gr_high_q, vsh_cut, phie_cut, sw_cut, rt_cut,
    )
    interpreted = integrate_show_intervals(interpreted, shows_by_well.get(well, pd.DataFrame()))
    interpreted = apply_historical_test_overlay(interpreted, well)
    models[well] = apply_unsupervised(interpreted, clusters, contamination)
    calibration[well] = params

# Overview KPIs
summary_rows = []
for well, df in models.items():
    s = thickness_summary(df).iloc[0].to_dict()
    s["Well"] = well
    s["Top depth (m)"] = df["DEPTH"].min()
    s["Base depth (m)"] = df["DEPTH"].max()
    s["GR clean"] = calibration[well]["GR_clean"]
    s["GR shale"] = calibration[well]["GR_shale"]
    summary_rows.append(s)
summary = pd.DataFrame(summary_rows).set_index("Well")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Wells", len(models))
c2.metric("Combined logged interval", f"{summary['Gross logged (m)'].sum():,.0f} m")
c3.metric("Combined net reservoir", f"{summary['Net reservoir (m)'].sum():,.1f} m")
c4.metric("Combined net pay", f"{summary['Net pay (m)'].sum():,.1f} m")

pages = st.tabs(["Field overview", "Well interpretation", "Mud gas & fluid model", "Formation calibration", "Correlation", "Crossplots & ML", "Sensitivity", "Downloads & methodology"])

with pages[0]:
    st.subheader("Field summary")
    display_summary = summary.copy()
    for c in ["N/G reservoir", "N/G pay", "Mean PHIE in pay", "Mean Sw in pay"]:
        display_summary[c] = display_summary[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    st.dataframe(display_summary.round(3), use_container_width=True)
    long = summary.reset_index().melt(id_vars="Well", value_vars=["Net reservoir (m)", "Net pay (m)"], var_name="Class", value_name="Thickness_m")
    st.plotly_chart(px.bar(long, x="Well", y="Thickness_m", color="Class", barmode="group", title="Net thickness comparison"), use_container_width=True)
    st.info("K-1 is merged from two overlapping SLAM runs. K-5 uses the merged PEX-HRLA-HNGS-DSI-PPC-SP LAS and can accept replacement LAS/CSV uploads.")
    st.caption(f"Current saturation model: {saturation_model}. Historical K-1 reference used a deterministic Dual Water Model; the in-app Dual-Water option is explicitly a screening approximation unless calibrated bound-water parameters are supplied.")

with pages[1]:
    selected = st.selectbox("Select well", list(models.keys()))
    full = models[selected]
    dmin, dmax = float(full["DEPTH"].min()), float(full["DEPTH"].max())
    interval = st.slider("Displayed depth interval (m)", dmin, dmax, (dmin, dmax))
    view = full[full["DEPTH"].between(interval[0], interval[1])]
    st.plotly_chart(log_panel(view, f"{selected} — interpreted log panel"), use_container_width=True)
    zones = contiguous_zones(full)
    st.subheader("Detected pay zones")
    if zones.empty:
        st.warning("No interval passes the current cutoffs.")
    else:
        st.dataframe(zones.round({"Top_m": 2, "Base_m": 2, "Gross_m": 2, "Mean_PHIE": 3, "Mean_SW": 3, "Mean_RT": 2}), use_container_width=True)
    st.caption("The hydrocarbon-response label is a log-response screening class. Oil viscosity and light/heavy oil classification require PVT, viscosity, API gravity, NMR mobility or valid test/sample data.")

with pages[2]:
    st.subheader("Integrated mud-gas chromatography & fluid-character model")
    st.caption("Primary fluid typing uses Haworth Wetness–Balance–Character. Pixler C1/C2 behavior, gas strength and fluorescence are independent confidence checks. Heavy-oil labels are screening tendencies only.")
    gas_well = st.selectbox("Gas-model well", list(models.keys()), key="gaswell")
    gfull = models[gas_well]
    shows = shows_by_well.get(gas_well, pd.DataFrame())
    if shows.empty:
        st.warning(
            "No valid Shows Summary data is loaded for this well. "
            "Upload the matching file in the sidebar under ‘Optional Shows Summary CSV files’ "
            "(for example: Azraq K-1_Shows Summary.csv). If deploying the full package, keep the data/ folder beside app.py."
        )
        st.info(f"Loaded Shows Summary wells: {', '.join(sorted(shows_by_well.keys())) if shows_by_well else 'none'}")
    else:
        sig_floor = st.number_input("Significant-show TG threshold (ppm)", 100.0, 10000.0, 1000.0, 100.0)
        sig = shows[shows["TG"] >= sig_floor].copy()
        gc1,gc2,gc3,gc4 = st.columns(4)
        gc1.metric("Recorded shows", len(shows))
        gc2.metric(f"TG ≥ {sig_floor:,.0f} ppm", len(sig))
        gc3.metric("Maximum TG", f"{shows.TG.max():,.0f} ppm")
        gc4.metric("Median wetness (significant)", f"{sig.WETNESS.median():.1f}%" if not sig.empty else "—")
        dmin,dmax=float(gfull.DEPTH.min()),float(gfull.DEPTH.max())
        grange=st.slider("Integrated depth interval (m)",dmin,dmax,(dmin,dmax),key="gasdepth")
        gv=gfull[gfull.DEPTH.between(*grange)]
        st.plotly_chart(gas_log_panel(gv, f"{gas_well} — petrophysics + mud-gas fluid screen"), use_container_width=True)
        st.subheader("Significant shows and model classification")
        cols=["SHOW_NO","FROM_M","TO_M","TG","C1","C2","C3","C4","C5","WETNESS","BALANCE","CHARACTER","C1_C2","HAWORTH_CLASS","PIXLER_CLASS","FLUID_CLASS","CONFIDENCE_SCORE","CONFIDENCE","DIRECT_FLUOR","CUT_FLUOR"]
        st.dataframe(sig[cols].sort_values("TG",ascending=False).round(2), use_container_width=True)
        if not sig.empty:
            fsum=sig.copy()
            fsum["BROAD_CLASS"]=fsum["FLUID_CLASS"].str.replace(" (low-signal)","",regex=False)
            counts=fsum.groupby("BROAD_CLASS").size().reset_index(name="Shows")
            st.plotly_chart(px.bar(counts,x="BROAD_CLASS",y="Shows",title="Significant-show fluid-character distribution"),use_container_width=True)
            st.plotly_chart(px.scatter(sig,x="WETNESS",y="BALANCE",size="TG",color="FLUID_CLASS",hover_data=["FROM_M","TO_M","C1_C2","CONFIDENCE_SCORE"],log_y=True,title="Wetness–Balance fluid-character crossplot"),use_container_width=True)
        st.warning("A 'Heavier-oil tendency' is not a measured heavy-oil/API-gravity result. C1–C5 gas chromatography does not quantify C6+, resins or asphaltenes; confirm with PVT/API gravity, viscosity, MDT/DST samples, NMR or well tests.")
        if gas_well == "Azraq K-1":
            st.success("K-1 includes historical DST calibration anchors in the Formation calibration tab. Use confirmed test fluid as the priority interpretation where intervals overlap.")
        if gas_well == "Azraq K-5":
            st.success("K-5 includes MDT fluid-sample and core calibration evidence. Formation-water-dominated MDT samples in SQ-2 take priority over a mud-gas or resistivity hydrocarbon screen at those exact depths.")

with pages[3]:
    st.subheader("K-1 / K-5 formation and fluid calibration")
    st.caption("Observed test evidence is displayed separately from model predictions. Where valid DST/PVT evidence exists, it should take interpretive priority over a mud-gas screening label.")
    k1 = models.get("Azraq K-1", pd.DataFrame())
    if k1.empty:
        st.warning("Azraq K-1 is not loaded.")
    else:
        ft = formation_tops.get("Azraq K-1", pd.DataFrame()).copy()
        if not ft.empty:
            st.markdown("#### Formation framework and salinity context")
            showcols=[c for c in ["FORMATION","TOP_M","BASE_M","SALINITY_MIN_PPM","SALINITY_MAX_PPM","RW_OHM_M"] if c in ft.columns]
            st.dataframe(ft[showcols], use_container_width=True)
            st.info("Historical salinity ranges are contextual evidence only. The app does not automatically convert salinity to Rw because temperature and water-composition calibration are not supplied.")

        st.markdown("#### Historical DST / fluid calibration anchors")
        st.dataframe(HISTORICAL_TESTS, use_container_width=True)
        validation = model_test_validation(k1, "Azraq K-1")
        st.markdown("#### Model-versus-test validation")
        if validation.empty:
            st.info("No validation intervals could be evaluated.")
        else:
            st.dataframe(validation.round({"API":2,"Median PHIE":3,"Median Sw":3,"Pay fraction":2,"Max TG":1,"Median Wetness":1}), use_container_width=True)
            agree = (validation["Agreement"]=="Compatible").sum()
            st.metric("Compatible model/test intervals", f"{agree}/{len(validation)}")
            plotv=validation.copy()
            plotv["Mid_m"]=(plotv["Top_m"]+plotv["Base_m"])/2
            st.plotly_chart(px.scatter(plotv,x="API",y="Mid_m",color="Observed",symbol="Agreement",hover_data=["Test","Formation","Model fluid","Rate/Result"],title="Historical fluid-test calibration anchors (API where measured)"),use_container_width=True)

        st.markdown("#### Historical pressure-gradient evidence")
        st.dataframe(PRESSURE_GRADIENTS, use_container_width=True)
        pg=PRESSURE_GRADIENTS.copy(); pg["MID_M"]=(pg.TOP_M+pg.BASE_M)/2
        st.plotly_chart(px.scatter(pg,x="GRADIENT_PSI_FT",y="MID_M",color="INTERPRETATION",hover_data=["FORMATION","TOP_M","BASE_M"],title="K-1 pressure gradients from historical evaluation"),use_container_width=True)

        st.markdown("#### Formation-level current model summary")
        rows=[]
        for fm,g in k1[k1.FORMATION.ne("Unassigned")].groupby("FORMATION"):
            rows.append({"Formation":fm,"Top_m":g.DEPTH.min(),"Base_m":g.DEPTH.max(),"Mean Rw used":g.RW_USED.mean(),"Net reservoir m":thickness_summary(g)["Net reservoir (m)"].iloc[0],"Net pay m":thickness_summary(g)["Net pay (m)"].iloc[0],"Mean PHIE":g.PHIE.mean(),"Mean Sw":g.SW.mean(),"Max TG":g.TG.max(),"Dominant model fluid":g.FLUID_CLASS.replace("",np.nan).dropna().mode().iloc[0] if not g.FLUID_CLASS.replace("",np.nan).dropna().mode().empty else "—"})
        if rows: st.dataframe(pd.DataFrame(rows).round(3),use_container_width=True)

    st.markdown("---")
    st.markdown("### Azraq K-5 calibration evidence")
    k5 = models.get("Azraq K-5", pd.DataFrame())
    if k5.empty:
        st.info("K-5 log data are not loaded.")
    else:
        k5ft = formation_tops.get("Azraq K-5", pd.DataFrame())
        if not k5ft.empty:
            st.markdown("#### K-5 formation framework")
            st.dataframe(k5ft[[c for c in ["FORMATION","TOP_M","BASE_M"] if c in k5ft.columns]], use_container_width=True)
        st.markdown("#### K-5 MDT fluid samples")
        st.dataframe(K5_FLUID_SAMPLES, use_container_width=True)
        st.warning("The three available SQ-2 MDT samples are formation-water dominated; they do not provide a measured API gravity for K-5 oil.")
        st.markdown("#### K-5 conventional cores")
        st.dataframe(K5_CORING, use_container_width=True)
        rows=[]
        for fm,g in k5[k5.FORMATION.ne("Unassigned")].groupby("FORMATION"):
            ts=thickness_summary(g).iloc[0]
            rows.append({"Formation":fm,"Top_m":g.DEPTH.min(),"Base_m":g.DEPTH.max(),"Net reservoir m":ts["Net reservoir (m)"],"Net pay m":ts["Net pay (m)"],"Mean PHIE":g.PHIE.mean(),"Mean Sw":g.SW.mean(),"Max TG":g.TG.max(),"Dominant model fluid":g.FLUID_CLASS.replace("",np.nan).dropna().mode().iloc[0] if not g.FLUID_CLASS.replace("",np.nan).dropna().mode().empty else "—"})
        if rows:
            st.markdown("#### K-5 formation-level model summary")
            st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)

with pages[4]:
    corr_curve = st.selectbox("Correlation curve", ["GR", "RT", "PHIE", "SW", "DT", "RHOB"])
    st.plotly_chart(correlation_panel(models, corr_curve), use_container_width=True)
    st.caption("This is a depth-aligned comparison. True stratigraphic correlation requires formation tops, markers and preferably TVDSS coordinates.")

with pages[5]:
    selected_ml = st.selectbox("ML well", list(models.keys()), key="mlwell")
    mdf = models[selected_ml]
    xcol = st.selectbox("X", ["GR", "RT", "RHOB", "NPHI", "PHIE", "SW"], index=0)
    ycol = st.selectbox("Y", ["PHIE", "SW", "RT", "RHOB", "NPHI", "DT"], index=0)
    plot_df = mdf[[xcol, ycol, "FACIES_CLUSTER", "PAY", "ANOMALY", "DEPTH"]].dropna(subset=[xcol, ycol]).copy()
    plot_df["FACIES_CLUSTER"] = plot_df["FACIES_CLUSTER"].astype("Int64").astype(str)
    st.plotly_chart(px.scatter(plot_df, x=xcol, y=ycol, color="FACIES_CLUSTER", symbol="PAY", hover_data=["DEPTH", "ANOMALY"], title="Unsupervised electrofacies crossplot"), use_container_width=True)
    anomaly_count = int(mdf["ANOMALY"].sum())
    st.write(f"Isolation Forest flagged **{anomaly_count:,}** samples for QC review. These may represent tool problems, washouts, spikes or genuinely unusual geology.")

with pages[6]:
    st.subheader("Rw sensitivity")
    sens_well = st.selectbox("Sensitivity well", list(models.keys()), key="senswell")
    rw_values = np.linspace(max(0.01, rw * 0.5), rw * 1.5, 9)
    rows = []
    base = wells[sens_well]
    for test_rw in rw_values:
        temp, _ = calculate_petrophysics(base, sens_well, formation_tops, test_rw, {}, a, m, n, rho_ma, rho_fl, shale_model, saturation_model, rshale, rwb, shale_bound_phi, gr_low_q, gr_high_q, vsh_cut, phie_cut, sw_cut, rt_cut)
        s = thickness_summary(temp).iloc[0]
        rows.append({"Rw": test_rw, "Net pay (m)": s["Net pay (m)"], "Mean Sw in pay": s["Mean Sw in pay"]})
    sens = pd.DataFrame(rows)
    st.plotly_chart(px.line(sens, x="Rw", y="Net pay (m)", markers=True, title="Net pay sensitivity to Rw"), use_container_width=True)
    st.dataframe(sens.round(3), use_container_width=True)

with pages[7]:
    st.subheader("Download interpreted results")
    for well, df in models.items():
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(f"Download {well} interpreted CSV", csv, file_name=f"{well.replace(' ', '_')}_interpreted.csv", mime="text/csv")
    st.download_button("Download field summary", summary.reset_index().to_csv(index=False).encode("utf-8"), file_name="Azraq_field_summary.csv", mime="text/csv")
    for well, sh in shows_by_well.items():
        st.download_button(f"Download {well} gas-ratio & fluid-class table", sh.to_csv(index=False).encode("utf-8"), file_name=f"{well.replace(' ','_')}_gas_fluid_model.csv", mime="text/csv")
    st.download_button("Download historical K-1 DST calibration table", HISTORICAL_TESTS.to_csv(index=False).encode("utf-8"), file_name="Azraq_K-1_historical_DST_calibration.csv", mime="text/csv")
    st.download_button("Download K-1 model-vs-test validation", model_test_validation(models.get("Azraq K-1", pd.DataFrame()), "Azraq K-1").to_csv(index=False).encode("utf-8"), file_name="Azraq_K-1_model_vs_test_validation.csv", mime="text/csv")
    st.download_button("Download K-5 MDT fluid calibration", K5_FLUID_SAMPLES.to_csv(index=False).encode("utf-8"), file_name="Azraq_K-5_MDT_fluid_calibration.csv", mime="text/csv")
    st.download_button("Download K-5 core calibration", K5_CORING.to_csv(index=False).encode("utf-8"), file_name="Azraq_K-5_core_calibration.csv", mime="text/csv")
    st.markdown("""
    **Workflow included**
    1. Curve mnemonic harmonization across SLAM, PEX and K-5 LAS inputs.
    2. Quality-based merge of overlapping K-1 runs.
    3. Quantile-based GR endpoints and selectable Vsh model.
    4. Density-neutron porosity plus selectable Archie, Simandoux, Indonesian and Dual-Water screening saturation models.
    5. Formation-aware K-1 and K-5 frameworks, optional formation-specific Rw, and historical PHIE/Sw cutoff preset.
    6. Reservoir/pay cutoffs, gap-safe net thickness and contiguous pay-zone extraction.
    7. Haworth Wetness–Balance–Character and Pixler C1/C2 mud-gas fluid screening.
    8. Historical K-1 DST/API/pressure calibration plus K-5 MDT/core calibration evidence.
    9. Depth integration, K-means electrofacies, anomaly screening and Rw sensitivity.

    **Evidence hierarchy:** DST/PVT/fluid samples > calibrated MDT/FMT pressure > calibrated petrophysics > mud-gas ratios > raw log-response screen.

    **Calibration priorities:** unified TVDSS across K-1/K-2/K-3/K-5, K-5 API/PVT/viscosity where obtainable, K-2/K-3 calibrated tests,, measured formation-specific Rw and temperature correction, core/SCAL, NMR, MDT/RFT pressure, PVT/API/viscosity and well-test results.
    """)
