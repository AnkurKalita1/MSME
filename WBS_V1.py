import os
import json
import uuid
import warnings
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import boto3
from decimal import Decimal
from sklearn.preprocessing import MultiLabelBinarizer

def ddb_to_python(value):
    if isinstance(value, list):
        return [ddb_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: ddb_to_python(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return float(value)
    return value

def load_boq_dataset_from_dynamodb(table_name="MSME_BOQ_WBS"):
    print(" Loading BOQ from DynamoDB...")
    dynamodb = boto3.resource(
        "dynamodb",
        region_name="ap-south-1",
        endpoint_url="http://localhost:8000",
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy"
    )
    table = dynamodb.Table(table_name)
    response = table.scan()
    items = response.get("Items", [])
    clean_items = [ddb_to_python(i) for i in items]
    df = pd.DataFrame(clean_items)
    df["boq_text"] = df["boq_text"].astype(str)
    if "subcategory" not in df.columns:
        df["subcategory"] = "general"
    df["subcategory"] = df["subcategory"].fillna("general").astype(str).apply(normalize_subcat_name)
    for col in ["quantity", "rate_most", "rate_min", "rate_max", "confidence"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].apply(lambda x: float(x) if x not in [None, ""] else 0.0)
    df["tasks_list"] = df["tasks_list"].apply(lambda x: [normalize_text(t) for t in x])
    df["stages_list"] = df["stages_list"].apply(lambda x: [int(s) for s in x])
    print(f" Loaded {len(df)} rows from DynamoDB.")
    return df

HOURS_PER_DAY = 8.0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

TASK_DURATION_BASE_HOURS = {
    "site measurement": 2.0, "layout marking": 3.0, "surface preparation": 6.0, "putty application": 4.0,
    "primer application": 3.0, "paint coat 1": 3.0, "paint coat 2": 3.0, "finishing": 8.0, "touchup": 2.0,
    "adhesive application": 4.0, "tile laying": 12.0, "grouting": 6.0, "edge finishing": 4.0,
    "frame fabrication": 10.0, "panel installation": 6.0, "screw fixing": 2.0, "board installation": 6.0,
    "channel fixing": 4.0, "joint taping": 3.0, "pipe cutting": 2.0, "pipe installation": 6.0,
    "joint sealing": 3.5, "pressure testing": 3.0, "conduit installation": 6.0, "cable pulling": 6.0,
    "db fabrication": 8.0, "testing and earthing": 2.5, "sludge removal": 4.0, "scrubbing": 4.0,
    "chemical cleaning": 6.0, "rinsing": 2.0, "disinfection": 2.0, "drying": 1.0, "excavation": 12.0,
    "rebar placement": 10.0, "formwork setup": 10.0, "concrete pouring": 12.0, "curing": 16.0,
    "__default__": 6.0
}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().replace("_", " ").split())

def normalize_subcat_name(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lower().replace(" ", "")

def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_stages(cell: str) -> List[int]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    v = safe_json_load(s)
    if isinstance(v, list):
        try:
            return [int(x) for x in v]
        except:
            pass
    s2 = s.replace("[","").replace("]","").replace('"',"")
    parts = [p.strip() for p in s2.split(",") if p.strip().isdigit()]
    return [int(p) for p in parts]

def parse_tasks(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    s = str(cell)
    v = safe_json_load(s)
    if isinstance(v, list):
        return [normalize_text(x) for x in v]
    if ";" in s:
        return [normalize_text(x) for x in s.split(";") if x.strip()]
    if "," in s:
        return [normalize_text(x) for x in s.split(",") if x.strip()]
    return [normalize_text(s)]

def generate_fixed_stages(n_tasks: int) -> List[int]:
    stages = []
    for i in range(n_tasks):
        if i == 0: stages.append(1)
        elif i == 1: stages.append(2)
        elif i == n_tasks - 2: stages.append(4)
        elif i == n_tasks - 1: stages.append(5)
        else: stages.append(3)
    return stages

def load_boq_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "boq_text" not in df.columns or "tasks" not in df.columns or "stages" not in df.columns:
        raise ValueError("CSV must contain columns: boq_text, tasks, stages")
    df["boq_text"] = df["boq_text"].fillna("").astype(str)
    if "subcategory" not in df.columns:
        df["subcategory"] = "general"
    else:
        df["subcategory"] = df["subcategory"].astype(str).str.strip().str.lower().str.replace(" ", "")
    for c in ["region","season","project_type","quantity","unit","grade",
              "rate_most","rate_min","rate_max","confidence"]:
        if c not in df.columns:
            df[c] = np.nan
    df["quantity"] = pd.to_numeric(df["quantity"].fillna(1), errors="coerce").fillna(1.0)
    df["region"] = df["region"].fillna("central").astype(str)
    df["season"] = df["season"].fillna("normal").astype(str)
    df["project_type"] = df["project_type"].fillna("Commercial").astype(str)
    df["grade"] = df["grade"].fillna("B").astype(str)
    df["tasks_list"] = df["tasks"].apply(parse_tasks)
    df["stages_list"] = df["stages"].apply(parse_stages)
    df = df.drop_duplicates(subset=["boq_text"], keep="first").reset_index(drop=True)
    for i, r in df.iterrows():
        if len(r["tasks_list"]) != len(r["stages_list"]):
            df.at[i, "stages_list"] = generate_fixed_stages(len(r["tasks_list"]))
    print("Final BOQ rows:", len(df))
    return df

def build_subcat_task_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    mapping = {}
    for _, r in df.iterrows():
        sub = r["subcategory"]
        if sub not in mapping:
            mapping[sub] = set()
        mapping[sub].update(r["tasks_list"])
    return {k: sorted(list(v)) for k,v in mapping.items()}

def detect_subcategory(boq_text: str, subcat_map: Dict[str, List[str]]) -> Optional[str]:
    text = (boq_text or "").lower()
    for sub in subcat_map.keys():
        if sub in text:
            return sub
    tiling_kw = ["tile","tiles","tiling","vitrified","ceramic","granite","marble","porcelain",
                 "stone tiles","mosaic tiles","wall tiles","floor tiles","toilet tiles","skirting",
                 "staircase tiles","grout","grouting","adhesive","tile adhesive","tile spacer",
                 "flooring","leveling","floor leveling","tile cutting","wet mix"]
    if any(w in text for w in tiling_kw):
        return "tiling"
    painting_kw = [
        "paint", "painting", "primer", "putty", "emulsion", "acrylic", "coat", "coats",
        "finish coat", "touchup", "distemper", "weather coat", "surface preparation",
        "sanding", "exterior", "external", "weather shield", "weatherproof",
        "patch repair", "crack filling", "wall painting", "waterproof"
    ]
    if any(w in text for w in painting_kw):
        return "painting"
    ceiling_kw = ["gypsum","false ceiling","ceiling board","grid ceiling","ceilings","plaster of paris",
                  "pop","channel fixing","screw fixing","perimeter channel","gi channel","ceiling tile"]
    if any(w in text for w in ceiling_kw):
        return "falseceiling"
    piping_kw = ["pipe","piping","plumbing","upvc","cpvc","gi pipe","joint sealing","pressure testing",
                 "pipe cutting","drain line","water supply","sewer","waste pipe","valve installation",
                 "tapping","pipe laying","pipe joint","concealed piping"]
    if any(w in text for w in piping_kw):
        return "piping"
    electrical_kw = [
        "wiring","cable","conduit","earthing","mccb","db","electrical","switch","socket",
        "junction box","panel board","breaker","cabling","light points","fixtures",
        "distribution board","mcbox","cable tray","gi conduit","fan point","lamp installation"
    ]
    if any(w in text for w in electrical_kw):
        return "wiring"
    tank_kw = [
        "sump","tank","cleaning","desludging","disinfection","chlorination","scrubbing",
        "underground sump","tank brushing","tank washing","tank drainage",
        "algae removal","bacterial cleaning"
    ]
    if any(w in text for w in tank_kw):
        if "tankcleaning" in subcat_map:
            return "tankcleaning"
        if "tank" in subcat_map:
            return "tank"
        return "tank"
    return None

def prepare_task_training(df: pd.DataFrame):
    X = df["boq_text"].astype(str).tolist()
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["tasks_list"])
    return X, Y, mlb

def train_task_classifier(X, Y):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_vec = vec.fit_transform(X)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=3000), n_jobs=-1)
    clf.fit(X_vec, Y)
    return {"vectorizer": vec, "classifier": clf}

def prepare_stage_training(df: pd.DataFrame):
    rows=[]
    for _,r in df.iterrows():
        for t,st in zip(r["tasks_list"], r["stages_list"]):
            rows.append((t, st))
    X =[x[0] for x in rows]
    y =[x[1] for x in rows]
    return X,y

def train_stage_classifier(X, y):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_vec = vec.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_vec, y, test_size=0.15, random_state=RANDOM_SEED
    )

    clf = LogisticRegression(max_iter=3000, multi_class="auto")
    clf.fit(X_train, y_train)

    acc = clf.score(X_val, y_val)
    print("Stage Accuracy:", acc)

    return {
        "vectorizer": vec,
        "classifier": clf,
        "val_acc": acc
    }

def compute_synthetic_task_duration_days(task_name: str, qty: float, region: str, season: str, grade: str,
                                         rate_most: float, rate_min: float, rate_max: float, confidence: float) -> float:
    task = normalize_text(task_name)
    BASE = TASK_DURATION_BASE_HOURS.get(task, TASK_DURATION_BASE_HOURS["__default__"])
    BASE = {
        "tile laying": 6.0,
        "grouting": 3.0,
        "adhesive application": 2.0,
        "edge finishing": 2.0,
        "finishing": 4.0
    }.get(task, BASE)
    qty_factor = 1.0 + np.clip(qty / 100.0, 0, 1.5)
    season = (season or "normal").lower()
    season_mul = {
        "normal": 1.0,
        "monsoon": 1.05,
        "festival": 1.03,
        "winter": 0.98
    }.get(season, 1.0)
    region_mul = {
        "central": 1.0,
        "south": 1.01,
        "north": 1.02,
        "east": 1.01,
        "west": 1.01
    }.get(region.lower(), 1.0)
    grade_mul = {
        "a": 0.97,
        "b": 1.00,
        "c": 1.03
    }.get(str(grade).lower(), 1.0)
    seed = abs(hash(task)) % 1000
    rnd_small = (seed % 7) / 50.0
    rnd_tiny = (seed % 3) / 200.0
    total_variation = 1.0 + rnd_small + rnd_tiny
    hours = BASE * qty_factor * season_mul * region_mul * grade_mul * total_variation
    return max(hours / HOURS_PER_DAY, 0.01)

def prepare_duration_training(df: pd.DataFrame):
    rows=[]
    for _,r in df.iterrows():
        qty = float(r.get("quantity",1.0) or 1.0)
        region = r.get("region","central")
        season = r.get("season","normal")
        grade = r.get("grade","B")
        rate_most = float(r.get("rate_most") if not pd.isna(r.get("rate_most")) else 0.0)
        rate_min = float(r.get("rate_min") if not pd.isna(r.get("rate_min")) else 0.0)
        rate_max = float(r.get("rate_max") if not pd.isna(r.get("rate_max")) else 0.0)
        confidence = float(r.get("confidence") if not pd.isna(r.get("confidence")) else 1.0)
        for t in r["tasks_list"]:
            dur_days = compute_synthetic_task_duration_days(t, qty, region, season, grade,
                                                            rate_most, rate_min, rate_max, confidence)
            rows.append({
                "task": t,
                "region": region,
                "season": season,
                "grade": grade,
                "log_qty": np.log1p(qty),
                "rate_most": rate_most,
                "rate_min": rate_min,
                "rate_max": rate_max,
                "confidence": confidence,
                "duration_days": dur_days
            })
    df_t = pd.DataFrame(rows)
    text_vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_txt = text_vec.fit_transform(df_t["task"])
    cat_cols = ["region","season","grade"]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = ohe.fit_transform(df_t[cat_cols])
    X_num = df_t[["log_qty","rate_most","rate_min","rate_max","confidence"]].fillna(0).values
    X = np.hstack([X_txt.toarray(), X_cat, X_num])
    y = df_t["duration_days"].values
    duration_meta = {"text_vec": text_vec, "ohe": ohe, "cat_cols": cat_cols,
                     "num_cols": ["log_qty","rate_most","rate_min","rate_max","confidence"]}
    return X, y, duration_meta

def train_duration_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    mape = mean_absolute_percentage_error(np.clip(y_val, 1e-6, None),
                                          np.clip(preds, 1e-6, None))
    print(f"Duration model Validation MAPE: {mape*100:.2f}%")
    return rf, float(mape)

def predict_tasks(boq_text: str, task_model: Dict[str,Any],
                  mlb: MultiLabelBinarizer, top_k:int=12):
    vec = task_model["vectorizer"].transform([boq_text])
    probs = task_model["classifier"].predict_proba(vec)[0]
    idx = np.argsort(probs)[::-1]
    tasks = [mlb.classes_[i] for i in idx[:top_k]]
    return [normalize_text(t) for t in tasks]

def duration_features_for_inference(task_name: str, qty: float, region: str, season: str, grade: str,
                                    rate_most: float, rate_min: float, rate_max: float, confidence: float,
                                    duration_meta: Dict[str,Any]):
    txt = duration_meta["text_vec"].transform([task_name]).toarray()
    ohe = duration_meta["ohe"]
    cat_cols = duration_meta["cat_cols"]
    cat_vals = [[region, season, grade]]
    X_cat = ohe.transform(cat_vals)
    X_num = np.array([[np.log1p(qty), float(rate_most or 0.0),
                       float(rate_min or 0.0), float(rate_max or 0.0),
                       float(confidence or 1.0)]])
    return np.hstack([txt, X_cat, X_num])

def predict_duration_pert(task_name, qty, region, season, grade,
                          rate_most, rate_min, rate_max, confidence,
                          duration_model, duration_meta):
    X = duration_features_for_inference(
        task_name, qty, region, season, grade,
        rate_most, rate_min, rate_max, confidence,
        duration_meta
    )
    most_days = float(duration_model.predict(X)[0])
    most_days = max(most_days, 0.01)
    optimistic_days = max(0.3, most_days * 0.70)
    pessimistic_days = max(optimistic_days, most_days * 1.40)
    expected_days = (optimistic_days + 4 * most_days + pessimistic_days) / 6.0
    return {
        "optimistic_hours": round(optimistic_days * HOURS_PER_DAY, 2),
        "most_likely_hours": round(most_days * HOURS_PER_DAY, 2),
        "pessimistic_hours": round(pessimistic_days * HOURS_PER_DAY, 2),
        "expected_hours": round(expected_days * HOURS_PER_DAY, 2)
    }

def generate_wbs(item: Dict[str, Any],
                 task_model: Dict[str, Any],
                 stage_model: Dict[str, Any],
                 duration_model,
                 duration_meta: Dict[str, Any],
                 mlb: MultiLabelBinarizer,
                 subcat_map: Dict[str, List[str]]) -> Dict[str, Any]:

    boq = item.get("boq_text", "")
    qty = float(item.get("quantity", 1.0) or 1.0)
    region = item.get("region", "central")
    season = item.get("season", "normal")
    grade = item.get("grade", "B")
    rate_most = float(item.get("rate_most") or 0.0)
    rate_min = float(item.get("rate_min") or 0.0)
    rate_max = float(item.get("rate_max") or 0.0)
    confidence = float(item.get("confidence") or 1.0)

    # ---------- Subcategory detection ----------
    input_subcat = item.get("subcategory")
    if input_subcat:
        input_subcat = normalize_subcat_name(input_subcat)

    detected_subcat = detect_subcategory(boq, subcat_map)
    subcat = detected_subcat or input_subcat or "general"
    subcat = normalize_subcat_name(subcat)

    allowed = subcat_map.get(subcat, [])

    # ---------- Predict tasks ----------
    predicted = predict_tasks(boq, task_model, mlb, top_k=20)

    # ---------- IMPROVEMENT 1: short-text fallback ----------
    if len(boq.strip().split()) <= 2:
        tasks = allowed.copy()
    else:
        tasks = [t for t in allowed if t in predicted]

    # ---------- IMPROVEMENT 2: poor classifier fallback ----------
    if len(tasks) < 3:
        tasks = allowed.copy()

    final = {"Planning": [], "Procurement": [], "Execution": [], "QC": [], "Billing": []}

    # ---------- Generate tasks ----------
    for t in tasks:
        orig_stage = int(stage_model["classifier"]
                         .predict(stage_model["vectorizer"].transform([t]))[0])

        bucket = "Planning" if orig_stage in [1, 2] else "Execution"
        new_stage = 1 if bucket == "Planning" else 3

        dur = predict_duration_pert(
            t, qty, region, season, grade,
            rate_most, rate_min, rate_max, confidence,
            duration_model, duration_meta
        )

        final[bucket].append({
            "task_id": str(uuid.uuid4())[:6],
            "task_name": t,
            "stage": new_stage,
            "duration": dur
        })

    # ---------- Procurement ----------
    def compute_procurement_duration(qty: float):
        base_hours = 1.5
        qty_factor = np.log1p(qty) * 0.35
        most = base_hours * (1 + qty_factor)
        optimistic = max(0.5, most * 0.70)
        pessimistic = most * 1.40
        expected = (optimistic + 4 * most + pessimistic) / 6.0
        return {
            "optimistic_hours": round(optimistic, 2),
            "most_likely_hours": round(most, 2),
            "pessimistic_hours": round(pessimistic, 2),
            "expected_hours": round(expected, 2)
        }

    final["Procurement"].append({
        "task_id": "PR001",
        "task_name": "material_procurement",
        "stage": 2,
        "duration": compute_procurement_duration(qty)
    })

    # ---------- QC ----------
    final["QC"].append({
        "task_id": "QC001",
        "task_name": "quality_check",
        "stage": 4,
        "duration": {
            "optimistic_hours": 0.8,
            "most_likely_hours": 1.5,
            "pessimistic_hours": 2.2,
            "expected_hours": round((0.8 + 4 * 1.5 + 2.2) / 6 * HOURS_PER_DAY, 2)
        }
    })

    # ---------- Billing ----------
    final["Billing"].append({
        "task_id": "BL001",
        "task_name": "final_billing",
        "stage": 5,
        "duration": {
            "optimistic_hours": 0.6,
            "most_likely_hours": 1.0,
            "pessimistic_hours": 1.6,
            "expected_hours": 1.03
        }
    })

    return {
        "boq_text": boq,
        "subcategory": subcat,
        "wbs": final,
        "wbs_id": str(uuid.uuid4())
    }


def parse_structured_input(text_or_dict: Any) -> Dict[str,Any]:
    out = {}
    if isinstance(text_or_dict, dict):
        out["boq_text"] = text_or_dict.get("boq_text") or text_or_dict.get("Material (description)") or text_or_dict.get("Material") or text_or_dict.get("description") or ""
        out["quantity"] = text_or_dict.get("quantity") or text_or_dict.get("Quantity") or text_or_dict.get("qty") or 1.0
        out["region"] = text_or_dict.get("region") or text_or_dict.get("Region") or "central"
        out["grade"] = text_or_dict.get("grade") or text_or_dict.get("Grade") or "B"
        out["season"] = text_or_dict.get("season") or text_or_dict.get("Season") or "normal"
        out["rate_most"] = text_or_dict.get("rate_most") or text_or_dict.get("Rate Most Likely") or 0.0
        out["rate_min"] = text_or_dict.get("rate_min") or text_or_dict.get("Rate Min") or 0.0
        out["rate_max"] = text_or_dict.get("rate_max") or text_or_dict.get("Rate Max") or 0.0
        out["confidence"] = text_or_dict.get("confidence") or text_or_dict.get("Confidence") or 1.0
        return out

    if isinstance(text_or_dict, str):
        lines = [l.strip() for l in text_or_dict.splitlines() if l.strip()]
        for line in lines:
            if ":" not in line:
                continue
            k,v = [x.strip() for x in line.split(":",1)]
            lk = k.lower()
            if "material" in lk or "description" in lk:
                out["boq_text"] = v
            elif "quantity" in lk or "qty" in lk:
                try: out["quantity"] = float(v)
                except: out["quantity"] = 1.0
            elif "region" in lk:
                out["region"] = v
            elif "grade" in lk:
                out["grade"] = v
            elif "season" in lk:
                out["season"] = v
            elif "rate most" in lk or "rate_most" in lk:
                try: out["rate_most"] = float(v)
                except: out["rate_most"] = 0.0
            elif "rate min" in lk or "rate_min" in lk:
                try: out["rate_min"] = float(v)
                except: out["rate_min"] = 0.0
            elif "rate max" in lk or "rate_max" in lk:
                try: out["rate_max"] = float(v)
                except: out["rate_max"] = 0.0
            elif "confidence" in lk:
                try: out["confidence"] = float(v)
                except: out["confidence"] = 1.0

        out.setdefault("boq_text","")
        out.setdefault("quantity",1.0)
        out.setdefault("region","central")
        out.setdefault("grade","B")
        out.setdefault("season","normal")
        out.setdefault("rate_most",0.0)
        out.setdefault("rate_min",0.0)
        out.setdefault("rate_max",0.0)
        out.setdefault("confidence",1.0)
        return out

    return {}

def main():
    warnings.filterwarnings("ignore")

    df = load_boq_dataset_from_dynamodb()
    subcat_map = build_subcat_task_map(df)

    X_text, Y, mlb = prepare_task_training(df)
    task_model = train_task_classifier(X_text, Y)

    X_stage, y_stage = prepare_stage_training(df)
    stage_model = train_stage_classifier(X_stage, y_stage)

    X_dur, y_dur, duration_meta = prepare_duration_training(df)
    duration_model, duration_mape = train_duration_model(X_dur, y_dur)

    print("\n==== MSME WBS PREDICTION ENGINE ====\n")

    boq_text = input("Enter BOQ description(eg:Underground sump cleaning upto 50KL): ").strip()
    qty = float(input("Quantity: ") or 1.0)
    region = input("Region (eg:central): ").strip() or "central"
    grade = input("Grade (eg:B): ").strip() or "B"
    season = input("Season (eg:normal): ").strip() or "normal"
    rate_most = float(input("Rate Most: ") or 0.0)
    rate_min = float(input("Rate Min: ") or 0.0)
    rate_max = float(input("Rate Max: ") or 0.0)
    confidence = float(input("Confidence (0–1): ") or 1.0)

    user_item = {
        "boq_text": boq_text,
        "quantity": qty,
        "region": region,
        "grade": grade,
        "season": season,
        "rate_most": rate_most,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "confidence": confidence
    }

    result = generate_wbs(
        user_item, task_model, stage_model,
        duration_model, duration_meta, mlb, subcat_map
    )

    print("\n\n========== WBS OUTPUT ==========\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nDuration Model MAPE:", round(duration_mape * 100, 2), "%")

if __name__ == "__main__":
    main()
