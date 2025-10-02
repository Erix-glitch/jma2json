# Based on JMA's official EEW Message Format Specification
# https://www.data.jma.go.jp/suishin/shiyou/pdf/no40202
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple, Union
import re
import json
from datetime import datetime
from pathlib import Path

# tables

TABLES_PATH = Path(__file__).with_name("tables.json")


def _load_builtin_tables(source: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    path = Path(source) if source else TABLES_PATH
    with open(path, "r", encoding="utf-8") as f:
        tables = json.load(f)
    if not isinstance(tables, dict):
        raise ValueError("tables.json must contain a JSON object with table mappings")
    return tables


_TABLES = _load_builtin_tables()

AA_TYPES = dict(_TABLES.get("AA_TYPES", {}))
OFFICES = dict(_TABLES.get("OFFICES", {}))
NN_TYPES = dict(_TABLES.get("NN_TYPES", {}))

# Minimal embedded epicenter codes (users can load full table).
# Keys as strings to preserve leading zeros if any.
EPICENTER_CODES = dict(_TABLES.get("EPICENTER_CODES", {}))

# Minimal region code examples for EBI (地域コード). Users can load full table at runtime.
REGION_CODES = dict(_TABLES.get("REGION_CODES", {}))

SHINDO_MAP = dict(_TABLES.get("SHINDO_MAP", {}))

def parse_int_or_none(s: str) -> Optional[int]:
    try:
        return int(s)
    except:
        return None

def parse_magnitude(mm: str) -> Optional[float]:
    if mm == "//":
        return None
    # "36" -> 3.6, "75" -> 7.5, "10" -> 1.0
    try:
        return float(mm[0] + "." + mm[1])
    except Exception:
        return None

def parse_latlon(token: str, is_lat: bool) -> Optional[float]:
    # N399 -> 39.9 ; E1409 -> 140.9
    if token in ("N///", "E////", "N//", "E//", "////", "///"):
        return None
    m = re.match(r"([NSEW])(\d+)", token)
    if not m: 
        return None
    hemi, val = m.groups()
    if not val:
        return None
    if len(val) >= 2:
        deg = float(val[:-1] + "." + val[-1])
    else:
        # fallback
        deg = float(val)
    if hemi in ("S",):
        deg = -deg
    # W longitudes negative if ever used
    if hemi in ("W",):
        deg = -deg
    return deg

def parse_cnf(cnf: str) -> Dict[str, Any]:
    """
    Cnf: C n f
    n: remaining including this (1-9 then A-Z as 10-35)
    f: 1=end, 0=continues
    """
    m = re.match(r"C([0-9A-Z])([01])$", cnf)
    if not m:
        return {"raw": cnf}
    n_ch, f = m.groups()
    # map A-Z to 10-35
    remainder = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(n_ch)
    return {"remainder_char": n_ch, "remainder": remainder, "is_final": f == "1"}

def decode_ncn(ncn: str) -> Dict[str, Any]:
    # NCNann where a:0 normal / 9 final ; nn: report number (01-99 or A0-Z9 for >99)
    m = re.match(r"NCN([0-9/])([0-9A-Z/]{2})$", ncn)
    res = {"raw": ncn, "final": None, "number": None}
    if not m:
        return res
    a, nn = m.groups()
    res["final"] = True if a == "9" else False if a == "0" else None
    res["number"] = nn
    return res

def decode_rk(rk: str) -> Dict[str, Any]:
    # RK n1 n2 n3 n4 n5
    m = re.match(r"RK([0-9/])([0-9/])([0-9/])([0-9/])([0-9/])$", rk)
    if not m:
        return {"raw": rk}
    n1, n2, n3, n4, n5 = m.groups()
    # Map descriptions per PDF
    epicenter_rel = {
        "1": "P/S level-cross, IPF 1pt, or assumed source (PLUM)",
        "2": "IPF 2pt",
        "3": "IPF 3-4pt",
        "4": "IPF 5+ pt",
        "5": "NIED system ≤4 pt or no precision info (Hi-net)",
        "6": "NIED system 5+ pt (Hi-net)",
        "7": "EPOS (offshore/outside network)",
        "8": "EPOS (inland/inside network)",
        "9": "Reserved",
        "/": "Unknown/Unset/Cancel",
    }
    depth_rel = epicenter_rel.copy()
    mag_rel = {
        "1": "Undefined",
        "2": "NIED system (Hi-net)",
        "3": "All P-phase",
        "4": "Mixed P/all-phase",
        "5": "All stations, all phases",
        "6": "EPOS",
        "7": "Undefined",
        "8": "P/S level-cross, or assumed source (PLUM)",
        "9": "Reserved",
        "/": "Unknown/Unset/Cancel",
    }
    m_used_pts = {
        "1": "1 point / level-cross / assumed (PLUM)",
        "2": "2 points",
        "3": "3 points",
        "4": "4 points",
        "5": "5+ points",
        "6": "Unused",
        "7": "Unused",
        "8": "Unused",
        "9": "Unused",
        "/": "Unknown/Unset/Cancel",
    }
    source_rel = {
        "1": "IPF 1pt or assumed source (PLUM)",
        "2": "IPF 2pt",
        "3": "IPF 3-4pt",
        "4": "IPF 5+ pt",
        "5": "Unused",
        "6": "Unused",
        "7": "Unused",
        "8": "Unused",
        "9": "Final-quality source; M/hypocenter no longer change (PLUM intensity may still change)",
        "/": "Unknown/Unset/Cancel",
    }
    return {
        "raw": rk,
        "n1_epicenter_reliability": epicenter_rel.get(n1, n1),
        "n2_depth_reliability": depth_rel.get(n2, n2),
        "n3_magnitude_reliability": mag_rel.get(n3, n3),
        "n4_magnitude_used_points": m_used_pts.get(n4, n4),
        "n5_source_reliability": source_rel.get(n5, n5),
    }

def decode_rt(rt: str) -> Dict[str, Any]:
    m = re.match(r"RT([0-9/])([0-9/])([0-9/])([0-9/])([0-9/])$", rt)
    if not m:
        return {"raw": rt}
    n1, n2, n3, n4, n5 = m.groups()
    return {
        "raw": rt,
        "n1_land_sea": {"0": "Land", "1": "Sea", "/": "Unknown/Unset"}.get(n1, n1),
        "n2_alert_level": {"0": "Forecast", "1": "Warning", "/": "Unknown/Unset"}.get(n2, n2),
        "n3_method": {"9": "PLUM-only valid (no source estimate in source+M method)", "/":"Unknown/Unset"}.get(n3, n3),
        "n4_reserved": n4,
        "n5_reserved": n5,
    }

def decode_rc(rc: str) -> Dict[str, Any]:
    m = re.match(r"RC([0-9/])([0-9/])([0-9/])([0-9/])([0-9/])$", rc)
    if not m:
        return {"raw": rc}
    n1, n2, n3, n4, n5 = m.groups()
    chg = {
        "0": "No/little change",
        "1": "Max predicted intensity increased by ≥1.0",
        "2": "Max predicted intensity decreased by ≥1.0",
        "/": "Unknown/Unset/Cancel",
    }
    reason = {
        "0": "No change",
        "1": "Mainly M changed (≥1.0)",
        "2": "Mainly source position changed (≥10.0 km)",
        "3": "Both M and source pos changed",
        "4": "Depth changed (≥30 km; otherwise none of above)",
        "9": "Changed due to PLUM prediction",
        "/": "Unknown/Unset/Cancel",
    }
    return {
        "raw": rc,
        "n1_change": chg.get(n1, n1),
        "n2_reason": reason.get(n2, n2),
        "n3_reserved": n3,
        "n4_reserved": n4,
        "n5_reserved": n5,
    }

def decode_shindo_token(token: str) -> Optional[str]:
    # "03" or "S6-//" etc in EBI blocks handled separately
    return SHINDO_MAP.get(token, token if token not in ("//",) else None)

def decode_ebi_block(tokens: List[str], kind: str) -> Tuple[Dict[str, Any], int]:
    """
    Parse one entry after EBI/ECI/EII.
    Format: fff Se1e2e3e4 hhmmss y1y2 (EBI)
            fffff ... (ECI)
            fffffff ... (EII)
    Returns (entry_dict, tokens_consumed)
    """
    idx = 0
    code = tokens[idx]; idx += 1

    S = tokens[idx]; idx += 1
    m = re.match(r"S([0-9/]{2}|5-|5\+|6-|6\+)([0-9/]{2}|5-|5\+|6-|6\+|//)$", S)
    if not m:
        # some forms like S0503 etc.
        m = re.match(r"S(.+)$", S)
    e1e2, e3e4 = None, None
    if m and len(m.groups()) >= 2:
        e1e2, e3e4 = m.groups()[0], m.groups()[1]
    else:
        # try to split e.g. "S0503" => "05","03"
        sraw = S[1:]
        if len(sraw) == 4:
            e1e2, e3e4 = sraw[:2], sraw[2:]
        elif len(sraw) == 2:
            e1e2, e3e4 = sraw, "//"

    time_tok = tokens[idx]; idx += 1
    y = tokens[idx]; idx += 1

    # interpret fields
    intensity_top = decode_shindo_token(e1e2) if e1e2 else None
    intensity_bottom = decode_shindo_token(e3e4) if e3e4 else None

    # hhmmss or "//////"
    t_h = t_m = t_s = None
    if re.match(r"^\d{6}$", time_tok):
        t_h, t_m, t_s = time_tok[:2], time_tok[2:4], time_tok[4:6]

    y1y2 = {"y1_alert": None, "y2_phase": None}
    if re.match(r"^[0-9/]{2}$", y):
        y1, y2 = y[0], y[1]
        y1y2["y1_alert"] = {"0":"Forecast","1":"Warning","/":"Unknown"}.get(y1, y1)
        y1y2["y2_phase"] = {
            "0":"Not yet arrived",
            "1":"Already arrived (predicted)",
            "9":"No arrival-time prediction (PLUM)",
            "/":"Unknown"
        }.get(y2, y2)

    label = {
        "EBI": "Region (地域)",
        "ECI": "Municipality (市町村)",
        "EII": "Station (観測点)",
    }.get(kind, "Unknown")

    return ({
        "scope_kind": label,
        "code": code,
        "name": REGION_CODES.get(code, None),
        "intensity_top": intensity_top,
        "intensity_bottom": intensity_bottom,  # if present, this is lower bound (range); if None, "以上"
        "arrival_time_hhmmss": f"{t_h}:{t_m}:{t_s}" if t_h else None,
        "flags": y1y2,
    }, idx)

@dataclass
class EEWMessage:
    raw: str
    aa: Optional[str] = None
    aa_desc: Optional[str] = None
    bb: Optional[str] = None
    bb_office: Optional[str] = None
    nn: Optional[str] = None
    nn_desc: Optional[str] = None
    transmit_time: Optional[str] = None  # YYYY-MM-DD HH:MM:SS (JST assumption)
    cnf: Dict[str, Any] = field(default_factory=dict)
    occurrence_time: Optional[str] = None
    nd_id: Optional[str] = None
    ncn: Dict[str, Any] = field(default_factory=dict)
    epicenter_code: Optional[str] = None
    epicenter_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_km: Optional[int] = None
    magnitude: Optional[float] = None
    max_predicted_intensity: Optional[str] = None
    rk: Dict[str, Any] = field(default_factory=dict)
    rt: Dict[str, Any] = field(default_factory=dict)
    rc: Dict[str, Any] = field(default_factory=dict)
    ebi: List[Dict[str, Any]] = field(default_factory=list)
    eci: List[Dict[str, Any]] = field(default_factory=list)
    eii: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

class EEWDecoder:
    def __init__(self):
        self.buffers: Dict[Tuple[str,str,str,str], Dict[str, Any]] = {}
        # key: (aa, bb, nn, transmit_time)

    def load_epicenter_table(self, mapping: Dict[str,str]):
        EPICENTER_CODES.update({str(k): v for k, v in mapping.items()})

    def load_region_table(self, mapping: Dict[str,str]):
        REGION_CODES.update({str(k): v for k, v in mapping.items()})

    @staticmethod
    def _fmt_time_yy(ts_yyMMddHHmmss: str) -> Optional[str]:
        # ts has 12 digits y y m m d d h h m m s s with y = last two digits of year
        if not re.match(r"^\d{12}$", ts_yyMMddHHmmss):
            return None
        yy = int(ts_yyMMddHHmmss[0:2])
        year = 2000 + yy if yy <= 69 else 1900 + yy  # heuristic; modern use 2000s
        month = int(ts_yyMMddHHmmss[2:4])
        day = int(ts_yyMMddHHmmss[4:6])
        hour = int(ts_yyMMddHHmmss[6:8])
        minute = int(ts_yyMMddHHmmss[8:10])
        second = int(ts_yyMMddHHmmss[10:12])
        try:
            return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        except:
            return None

    def _reassembly_key(self, aa: str, bb: str, nn: str, transmit_time: str) -> Tuple[str,str,str,str]:
        return (aa, bb, nn, transmit_time)

    def add_telegram(self, raw: str) -> Dict[str, Any]:
        """
        Add one raw telegram (string). Returns a dict with keys:
          - 'complete': bool (True when all parts reassembled)
          - 'messages': List[EEWMessage] (only when complete)
        """
        # Normalize whitespace; split at 9999= terminator chunk-wise
        cleaned = raw.strip().rstrip("=")
        # Tokenize on whitespace
        tokens = re.split(r"\s+", cleaned)

        # Find position of "9999"
        try:
            end_idx = tokens.index("9999")
            tokens = tokens[:end_idx]  # ignore trailing tokens after 9999
        except ValueError:
            # no terminator; attempt to proceed but mark incomplete
            pass

        # Parse header common fields
        if len(tokens) < 5:
            return {"complete": False, "error": "Too few tokens"}

        aa, bb, nn, ts, cnf = tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]
        cnf_info = parse_cnf(cnf)
        idx = 5

        msg = EEWMessage(raw=raw)
        msg.aa = aa; msg.aa_desc = AA_TYPES.get(aa, "Unknown")
        msg.bb = bb; msg.bb_office = OFFICES.get(bb, "Unknown")
        msg.nn = nn; msg.nn_desc = NN_TYPES.get(nn, "Unknown")
        msg.transmit_time = self._fmt_time_yy(ts)
        msg.cnf = cnf_info

        # The next token is usually occurrence/detection time (yoyomomododo...)
        if idx < len(tokens) and re.match(r"^\d{12}$", tokens[idx]):
            msg.occurrence_time = self._fmt_time_yy(tokens[idx]); idx += 1

        # Iterate remaining tokens
        while idx < len(tokens):
            tok = tokens[idx]

            if tok.startswith("ND"):
                msg.nd_id = tok.replace("ND", "")
                idx += 1
                continue
            if tok.startswith("NCN"):
                msg.ncn = decode_ncn(tok)
                idx += 1
                continue
            if tok.startswith("JD"):
                idx += 1; continue
            if tok.startswith("JN"):
                idx += 1; continue

            # Epicenter code kkk (3 digits or ///)
            if re.match(r"^\d{3}$", tok) or tok == "///":
                msg.epicenter_code = None if tok == "///" else tok
                msg.epicenter_name = EPICENTER_CODES.get(tok, None) if tok != "///" else None
                idx += 1
                # latitude, longitude, depth, magnitude, intensity may follow
                if idx < len(tokens):
                    lat = tokens[idx]; idx += 1
                    lon = tokens[idx]; idx += 1
                    dep = tokens[idx]; idx += 1
                    mag = tokens[idx]; idx += 1
                    inten = tokens[idx]; idx += 1

                    msg.latitude = parse_latlon(lat, True)
                    msg.longitude = parse_latlon(lon, False)
                    msg.depth_km = parse_int_or_none(dep) if dep not in ("///",) else None
                    msg.magnitude = parse_magnitude(mag)
                    msg.max_predicted_intensity = decode_shindo_token(inten)

                continue

            if tok.startswith("RK"):
                msg.rk = decode_rk(tok)
                idx += 1; continue

            if tok.startswith("RT"):
                msg.rt = decode_rt(tok)
                idx += 1; continue

            if tok.startswith("RC"):
                msg.rc = decode_rc(tok)
                idx += 1; continue

            if tok in ("EBI", "ECI", "EII"):
                kind = tok
                idx += 1
                # Repeated blocks until we hit a new label or end
                while idx < len(tokens):
                    # Stop if next token is a new label (two letters) or field like RK/RT/RC/9999
                    if re.match(r"^(EBI|ECI|EII|RK|RT|RC|ND|NCN|JD|JN)$", tokens[idx]):
                        break
                    # Need at least 4 tokens per entry
                    if idx + 3 >= len(tokens):
                        break
                    entry, used = decode_ebi_block(tokens[idx:], kind)
                    if kind == "EBI":
                        msg.ebi.append(entry)
                    elif kind == "ECI":
                        msg.eci.append(entry)
                    else:
                        msg.eii.append(entry)
                    idx += used
                continue

            # If unknown token, advance to prevent infinite loops
            idx += 1

        # Reassembly handling by (aa,bb,nn,transmit_time)
        key = self._reassembly_key(aa, bb, nn, ts)
        buf = self.buffers.setdefault(key, {"parts": [], "final_seen": False})
        buf["parts"].append(msg)
        if msg.cnf.get("is_final"):
            buf["final_seen"] = True

        # If final seen and the first part tells us how many, we can complete
        # Per PDF, order by descending 'n' remainder (first part has largest)
        # Here we just complete when final_seen is True and return all parts sorted
        if buf["final_seen"]:
            # Sort parts: by cnf remainder descending
            parts = sorted(buf["parts"], key=lambda m: m.cnf.get("remainder", 1), reverse=True)
            # Flatten EBI/ECI/EII across parts
            if len(parts) > 1:
                base = parts[0]
                for p in parts[1:]:
                    base.ebi.extend(p.ebi)
                    base.eci.extend(p.eci)
                    base.eii.extend(p.eii)
                # Return single consolidated message
                out = [base]
            else:
                out = parts
            # cleanup
            del self.buffers[key]
            return {"complete": True, "messages": out}

        return {"complete": False, "messages": []}

    def decode(self, raw: str) -> List[Dict[str, Any]]:
        """
        Convenience method for single-part telegrams.
        """
        res = self.add_telegram(raw)
        if res.get("complete"):
            return [m.to_dict() for m in res["messages"]]
        # if not complete, still return what we have
        return [m.to_dict() for m in self.buffers.get((), {}).get("parts", [])]

# ---- Utility to pretty print ----

def decode_to_json(raw: str, ensure_ascii=False, indent=2) -> str:
    dec = EEWDecoder()
    result = dec.add_telegram(raw)
    if result.get("complete"):
        payload = [m.to_dict() for m in result["messages"]]
    else:
        payload = []
    return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)

# ---- Optional: allow external JSON tables ----

def load_tables(decoder: EEWDecoder, epicenter_json_path: Optional[str]=None, region_json_path: Optional[str]=None):
    if epicenter_json_path:
        with open(epicenter_json_path, "r", encoding="utf-8") as f:
            decoder.load_epicenter_table(json.load(f))
    if region_json_path:
        with open(region_json_path, "r", encoding="utf-8") as f:
            decoder.load_region_table(json.load(f))
