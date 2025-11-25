# rules_monitoring.py
# Rule-based complication detector for intraoperative numeric monitors (prototype).
# INPUT: dict with keys:
#   'SpO2' (percent),
#   'MAP' (mmHg),
#   'HR' (bpm),
#   'EtCO2' (mmHg),
#   'CVP' (mmHg),
#   'Urine' (ml_per_hr),
#   'Temp' (Celsius),
#   'PI' (perfusion index, unitless)
#
# OUTPUT: one chosen complication (string), matched variant id, brief explanation of main contributing sensors.
#
# DISCLAIMER: prototype, not clinical-grade. Validate on real cases and consult clinicians.

from typing import Dict, List, Tuple, Any

# ---- helper ops ----
def check_cond(value, op: str, thresh) -> bool:
    if op == "<":   return value < thresh
    if op == "<=":  return value <= thresh
    if op == ">":   return value > thresh
    if op == ">=":  return value >= thresh
    if op == "==":  return value == thresh
    raise ValueError("Unknown op")

def eval_variant(reading: Dict[str, float], variant: List[Tuple[str, str, float]]) -> bool:
    # variant: list of (sensor_name, op, threshold)
    for (sensor, op, thr) in variant:
        if sensor not in reading:
            return False
        if not check_cond(reading[sensor], op, thr):
            return False
    return True

# ---- priority order for tie-breaking (first = highest priority) ----
PRIORITY = [
    "Hemodynamic shock",
    "Anaphylaxis / Allergic reaction",
    "Severe Hypoxia",
    "Myocardial infarction",
    "Jugular venous distension (JVD)",
    "Hypocoagulation",
    "Third-degree AV block",
    "Second-degree AV block",
    "First-degree AV block",
    "Atrial fibrillation (proxy)",
    "Cyanosis (isolated)",
    "Normal"
]

# ---- Rules DB: for each complication - list of variants
# Each variant is a list of 8 conditions: (sensor, op, threshold)
# Thresholds chosen to be clinically plausible for intraop monitoring (heuristics).

RULES = {
    # 1) Hypoxia (we label "Severe Hypoxia" for life-threatening patterns)
    "Severe Hypoxia": [
        # variant 1 - severe hypoxemia, preserved MAP
        [("SpO2","<=",85), ("MAP",">=",60), ("HR",">",50), ("EtCO2",">",30),
         ("CVP",">=",2), ("Urine",">=",10), ("Temp",">=",35.0), ("PI","<=",0.8)],
        # variant 2 - hypoventilation high EtCO2
        [("SpO2","<=",88), ("EtCO2",">=",55), ("MAP",">=",55), ("HR",">",40),
         ("CVP",">=",1), ("Urine",">=",5), ("Temp",">=",35.0), ("PI","<=",1.2)],
        # variant 3 - abrupt desaturation with low PI (poor perfusion)
        [("SpO2","<=",82), ("PI","<=",0.6), ("MAP",">=",50), ("HR",">=",60),
         ("EtCO2","<=",40), ("CVP",">=",0), ("Urine",">=",0), ("Temp",">=",34.5)],
        # variant 4 - moderate desaturation with hypotension component
        [("SpO2","<=",90), ("MAP","<=",65), ("HR",">=",55), ("EtCO2","<=",45),
         ("CVP",">=",1), ("Urine","<=",30), ("Temp","<=",36.5), ("PI","<=",1.0)],
        # variant 5 - low SpO2 with near-normal MAP but falling urine
        [("SpO2","<=",89), ("MAP",">=",65), ("HR",">",70), ("EtCO2","<=",50),
         ("CVP",">=",2), ("Urine","<=",25), ("Temp",">=",35.0), ("PI","<=",1.0)],
        # variant 6 - sudden fall of SpO2 with mild tachycardia
        [("SpO2","<=",86), ("HR",">=",90), ("MAP",">=",60), ("EtCO2","<=",40),
         ("CVP",">=",2), ("Urine",">=",5), ("Temp",">=",35.0), ("PI","<=",0.9)],
        # variant 7 - low SpO2 + low perfusion index, low urine (prolonged)
        [("SpO2","<=",88), ("PI","<=",0.7), ("Urine","<=",20), ("MAP",">=",55),
         ("HR",">=",50), ("EtCO2","<=",45), ("CVP",">=",1), ("Temp",">=",35.0)],
        # variant 8 - desat + hypercapnic pattern (retention)
        [("SpO2","<=",90), ("EtCO2",">=",60), ("MAP",">=",55), ("HR",">=",55),
         ("CVP",">=",0), ("Urine",">=",10), ("Temp",">=",35.0), ("PI","<=",1.5)],
        # variant 9 - severe desat + borderline MAP
        [("SpO2","<=",80), ("MAP","<=",70), ("HR",">",40), ("EtCO2","<=",45),
         ("CVP",">=",0), ("Urine","<=",20), ("Temp",">=",34.5), ("PI","<=",0.6)],
        # variant 10 - desat with low PI but normal EtCO2 (perfusion problem)
        [("SpO2","<=",88), ("PI","<=",0.5), ("EtCO2",">=",30), ("MAP",">=",60),
         ("HR",">=",60), ("CVP",">=",1), ("Urine",">=",10), ("Temp",">=",35.0)]
    ],

    # 2) Jugular venous distension (JVD) -> we use high CVP as main sign
    "Jugular venous distension (JVD)": [
        [("CVP",">=",12), ("MAP",">=",60), ("HR","<=",110), ("SpO2",">=",90),
         ("EtCO2",">=",30), ("Urine",">=",20), ("Temp",">=",35.0), ("PI",">=",0.5)],
        [("CVP",">=",10), ("MAP","<=",75), ("HR",">=",50), ("SpO2",">=",88),
         ("EtCO2","<=",50), ("Urine","<=",40), ("Temp",">=",35.0), ("PI","<=",2.5)],
        [("CVP",">=",14), ("MAP","<=",70), ("HR","<=",100), ("SpO2",">=",88),
         ("EtCO2","<=",45), ("Urine","<=",30), ("Temp",">=",35.0), ("PI","<=",1.8)],
        [("CVP",">=",11), ("MAP",">=",60), ("HR",">=",40), ("SpO2",">=",85),
         ("EtCO2",">=",25), ("Urine",">=",10), ("Temp",">=",35.0), ("PI",">=",0.4)],
        [("CVP",">=",13), ("MAP","<=",80), ("HR","<=",90), ("SpO2",">=",86),
         ("EtCO2","<=",50), ("Urine","<=",50), ("Temp",">=",35.0), ("PI","<=",3.0)],
        [("CVP",">=",12), ("MAP",">=",55), ("HR","<=",120), ("SpO2",">=",88),
         ("EtCO2",">=",30), ("Urine",">=",5), ("Temp",">=",34.5), ("PI","<=",2.0)],
        [("CVP",">=",15), ("MAP","<=",85), ("HR",">=",40), ("SpO2",">=",87),
         ("EtCO2","<=",45), ("Urine","<=",35), ("Temp",">=",35.0), ("PI","<=",1.2)],
        [("CVP",">=",10), ("MAP","<=",70), ("HR",">=",60), ("SpO2",">=",85),
         ("EtCO2",">=",30), ("Urine","<=",25), ("Temp",">=",35.0), ("PI","<=",1.0)],
        [("CVP",">=",11), ("MAP",">=",60), ("HR","<=",140), ("SpO2",">=",86),
         ("EtCO2","<=",55), ("Urine",">=",10), ("Temp",">=",35.0), ("PI",">=",0.3)],
        [("CVP",">=",12), ("MAP","<=",90), ("HR",">=",50), ("SpO2",">=",88),
         ("EtCO2",">=",28), ("Urine","<=",40), ("Temp",">=",35.0), ("PI","<=",5.0)]
    ],

    # 3) Hemodynamic shock (life-threatening)
    "Hemodynamic shock": [
        [("MAP","<=",50), ("HR",">=",110), ("SpO2","<=",92), ("EtCO2","<=",30),
         ("CVP","<=",5), ("Urine","<=",10), ("Temp","<=",35.5), ("PI","<=",0.6)],
        [("MAP","<=",55), ("HR",">=",100), ("SpO2","<=",90), ("EtCO2","<=",35),
         ("CVP","<=",4), ("Urine","<=",20), ("Temp","<=",36.0), ("PI","<=",0.8)],
        [("MAP","<=",60), ("HR",">=",120), ("SpO2","<=",94), ("EtCO2","<=",32),
         ("CVP","<=",3), ("Urine","<=",15), ("Temp","<=",36.0), ("PI","<=",1.0)],
        [("MAP","<=",50), ("HR","<=",45), ("SpO2","<=",90), ("EtCO2","<=",30),
         ("CVP",">=",10), ("Urine","<=",20), ("Temp","<=",36.0), ("PI","<=",0.7)],  # cardiogenic shock
        [("MAP","<=",55), ("HR",">=",90), ("SpO2","<=",92), ("EtCO2","<=",35),
         ("CVP","<=",6), ("Urine","<=",10), ("Temp","<=",36.0), ("PI","<=",0.9)],
        [("MAP","<=",45), ("HR",">=",130), ("SpO2","<=",88), ("EtCO2","<=",28),
         ("CVP","<=",2), ("Urine","<=",5), ("Temp","<=",35.0), ("PI","<=",0.5)],
        [("MAP","<=",58), ("HR",">=",100), ("SpO2","<=",90), ("EtCO2","<=",30),
         ("CVP","<=",4), ("Urine","<=",12), ("Temp","<=",36.0), ("PI","<=",0.7)],
        [("MAP","<=",50), ("HR",">=",95), ("SpO2","<=",90), ("EtCO2","<=",33),
         ("CVP","<=",5), ("Urine","<=",8), ("Temp","<=",36.0), ("PI","<=",0.6)],
        [("MAP","<=",60), ("HR",">=",110), ("SpO2","<=",92), ("EtCO2","<=",32),
         ("CVP","<=",3), ("Urine","<=",18), ("Temp","<=",35.5), ("PI","<=",0.8)],
        [("MAP","<=",55), ("HR",">=",105), ("SpO2","<=",91), ("EtCO2","<=",30),
         ("CVP","<=",4), ("Urine","<=",10), ("Temp","<=",36.5), ("PI","<=",0.9)]
    ],

    # 4) Hypocoagulation / severe bleeding (proxy — hemodynamic + low temp)
    "Hypocoagulation / Massive bleeding (proxy)": [
        [("MAP","<=",60), ("Urine","<=",10), ("SpO2","<=",92), ("HR",">=",110),
         ("CVP","<=",4), ("EtCO2","<=",30), ("Temp","<=",35.0), ("PI","<=",0.8)],
        [("MAP","<=",65), ("Urine","<=",15), ("HR",">=",100), ("SpO2","<=",94),
         ("CVP","<=",5), ("EtCO2","<=",35), ("Temp","<=",35.5), ("PI","<=",1.0)],
        [("MAP","<=",55), ("Urine","<=",5), ("HR",">=",120), ("SpO2","<=",90),
         ("CVP","<=",2), ("EtCO2","<=",28), ("Temp","<=",34.8), ("PI","<=",0.5)],
        [("MAP","<=",60), ("Urine","<=",8), ("HR",">=",95), ("SpO2","<=",92),
         ("CVP","<=",3), ("EtCO2","<=",32), ("Temp","<=",35.0), ("PI","<=",0.7)],
        [("MAP","<=",62), ("Urine","<=",12), ("HR",">=",100), ("SpO2","<=",93),
         ("CVP","<=",4), ("EtCO2","<=",34), ("Temp","<=",35.2), ("PI","<=",0.9)],
        [("MAP","<=",58), ("Urine","<=",6), ("HR",">=",110), ("SpO2","<=",90),
         ("CVP","<=",2), ("EtCO2","<=",30), ("Temp","<=",34.5), ("PI","<=",0.6)],
        [("MAP","<=",60), ("Urine","<=",10), ("HR",">=",105), ("SpO2","<=",91),
         ("CVP","<=",3), ("EtCO2","<=",33), ("Temp","<=",35.0), ("PI","<=",0.8)],
        [("MAP","<=",63), ("Urine","<=",12), ("HR",">=",98), ("SpO2","<=",92),
         ("CVP","<=",5), ("EtCO2","<=",34), ("Temp","<=",35.5), ("PI","<=",1.1)],
        [("MAP","<=",57), ("Urine","<=",7), ("HR",">=",115), ("SpO2","<=",89),
         ("CVP","<=",2), ("EtCO2","<=",29), ("Temp","<=",34.8), ("PI","<=",0.6)],
        [("MAP","<=",61), ("Urine","<=",9), ("HR",">=",100), ("SpO2","<=",92),
         ("CVP","<=",4), ("EtCO2","<=",32), ("Temp","<=",35.0), ("PI","<=",0.85)]
    ],

    # 5) Anaphylaxis / Allergic reaction (severe)
    "Anaphylaxis / Allergic reaction": [
        [("MAP","<=",55), ("HR",">=",100), ("SpO2","<=",92), ("EtCO2","<=",30),
         ("CVP","<=",6), ("Urine","<=",20), ("Temp","<=",37.5), ("PI","<=",0.9)],
        [("MAP","<=",60), ("HR",">=",110), ("SpO2","<=",90), ("EtCO2","<=",32),
         ("CVP","<=",5), ("Urine","<=",15), ("Temp","<=",37.0), ("PI","<=",0.8)],
        [("MAP","<=",50), ("HR",">=",120), ("SpO2","<=",88), ("EtCO2","<=",28),
         ("CVP","<=",4), ("Urine","<=",10), ("Temp","<=",36.5), ("PI","<=",0.6)],
        [("MAP","<=",58), ("HR",">=",105), ("SpO2","<=",92), ("EtCO2","<=",30),
         ("CVP","<=",6), ("Urine","<=",20), ("Temp","<=",37.0), ("PI","<=",1.0)],
        [("MAP","<=",55), ("HR",">=",95), ("SpO2","<=",91), ("EtCO2","<=",33),
         ("CVP","<=",6), ("Urine","<=",18), ("Temp","<=",37.0), ("PI","<=",1.2)],
        [("MAP","<=",52), ("HR",">=",110), ("SpO2","<=",89), ("EtCO2","<=",29),
         ("CVP","<=",5), ("Urine","<=",12), ("Temp","<=",36.8), ("PI","<=",0.7)],
        [("MAP","<=",54), ("HR",">=",100), ("SpO2","<=",90), ("EtCO2","<=",31),
         ("CVP","<=",6), ("Urine","<=",15), ("Temp","<=",37.2), ("PI","<=",1.0)],
        [("MAP","<=",56), ("HR",">=",115), ("SpO2","<=",89), ("EtCO2","<=",29),
         ("CVP","<=",4), ("Urine","<=",10), ("Temp","<=",36.6), ("PI","<=",0.8)],
        [("MAP","<=",60), ("HR",">=",105), ("SpO2","<=",92), ("EtCO2","<=",32),
         ("CVP","<=",5), ("Urine","<=",20), ("Temp","<=",37.0), ("PI","<=",1.0)],
        [("MAP","<=",55), ("HR",">=",120), ("SpO2","<=",90), ("EtCO2","<=",30),
         ("CVP","<=",5), ("Urine","<=",8), ("Temp","<=",36.8), ("PI","<=",0.6)]
    ],

    # 6) Cyanosis (isolated / peripheral)
    "Cyanosis (isolated)": [
        [("SpO2","<=",90), ("PI","<=",0.8), ("MAP",">=",60), ("HR",">=",50),
         ("EtCO2",">=",30), ("CVP",">=",1), ("Urine",">=",10), ("Temp",">=",35.0)],
        [("SpO2","<=",92), ("PI","<=",0.6), ("MAP",">=",55), ("HR",">=",60),
         ("EtCO2",">=",28), ("CVP",">=",0), ("Urine",">=",5), ("Temp",">=",34.5)],
        [("SpO2","<=",88), ("PI","<=",0.5), ("MAP",">=",50), ("HR",">=",50),
         ("EtCO2","<=",50), ("CVP",">=",1), ("Urine",">=",5), ("Temp",">=",35.0)],
        [("SpO2","<=",91), ("PI","<=",1.0), ("MAP",">=",60), ("HR",">=",55),
         ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",20), ("Temp",">=",35.0)],
        [("SpO2","<=",90), ("PI","<=",0.7), ("MAP",">=",55), ("HR",">=",60),
         ("EtCO2","<=",45), ("CVP",">=",1), ("Urine",">=",10), ("Temp",">=",35.0)],
        [("SpO2","<=",89), ("PI","<=",0.6), ("MAP",">=",50), ("HR",">=",70),
         ("EtCO2",">=",28), ("CVP",">=",1), ("Urine",">=",5), ("Temp",">=",35.0)],
        [("SpO2","<=",92), ("PI","<=",0.8), ("MAP",">=",60), ("HR",">=",48),
         ("EtCO2",">=",25), ("CVP",">=",0), ("Urine",">=",10), ("Temp",">=",35.5)],
        [("SpO2","<=",90), ("PI","<=",0.4), ("MAP",">=",55), ("HR",">=",50),
         ("EtCO2","<=",50), ("CVP",">=",1), ("Urine",">=",5), ("Temp",">=",34.8)],
        [("SpO2","<=",88), ("PI","<=",0.6), ("MAP",">=",60), ("HR",">=",60),
         ("EtCO2","<=",45), ("CVP",">=",0), ("Urine",">=",10), ("Temp",">=",35.0)],
        [("SpO2","<=",91), ("PI","<=",0.7), ("MAP",">=",55), ("HR",">=",55),
         ("EtCO2",">=",28), ("CVP",">=",1), ("Urine",">=",8), ("Temp",">=",35.0)]
    ],

    # 7) Normal
    "Normal": [
        [("SpO2",">=",95), ("MAP",">=",70), ("MAP","<=",100), ("HR",">=",55),
         ("HR","<=",100), ("EtCO2",">=",32), ("EtCO2","<=",45), ("CVP",">=",2)],
        # variants still must include all sensors; we put benign ranges
        [("SpO2",">=",96), ("MAP",">=",65), ("MAP","<=",105), ("HR",">=",50),
         ("HR","<=",105), ("EtCO2",">=",30), ("EtCO2","<=",48), ("CVP",">=",1)],
        [("SpO2",">=",95), ("PI",">=",0.8), ("MAP",">=",68), ("MAP","<=",100),
         ("HR",">=",55), ("HR","<=",95), ("Urine",">=",30), ("Temp",">=",36.0)],
        [("SpO2",">=",95), ("MAP",">=",70), ("HR",">=",55), ("HR","<=",100),
         ("EtCO2",">=",32), ("CVP",">=",1), ("Urine",">=",25), ("Temp",">=",36.0)],
        [("SpO2",">=",94), ("PI",">=",0.9), ("MAP",">=",65), ("HR",">=",50),
         ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",20), ("Temp",">=",36.0)],
        [("SpO2",">=",95), ("PI",">=",1.0), ("MAP",">=",70), ("HR",">=",55),
         ("EtCO2",">=",33), ("CVP",">=",2), ("Urine",">=",30), ("Temp",">=",36.0)],
        [("SpO2",">=",95), ("MAP",">=",68), ("HR",">=",55), ("EtCO2",">=",32),
         ("CVP",">=",1), ("Urine",">=",25), ("Temp",">=",36.0), ("PI",">=",0.8)],
        [("SpO2",">=",96), ("MAP",">=",70), ("HR",">=",60), ("EtCO2",">=",34),
         ("CVP",">=",2), ("Urine",">=",30), ("Temp",">=",36.0), ("PI",">=",1.0)],
        [("SpO2",">=",95), ("MAP",">=",65), ("HR",">=",50), ("EtCO2",">=",32),
         ("CVP",">=",1), ("Urine",">=",20), ("Temp",">=",36.0), ("PI",">=",0.7)],
        [("SpO2",">=",95), ("MAP",">=",70), ("HR",">=",55), ("EtCO2",">=",32),
         ("CVP",">=",1), ("Urine",">=",30), ("Temp",">=",36.0), ("PI",">=",0.8)]
    ],

    # 8) Myocardial infarction (proxy, hemodynamic signs without ECG/troponin)
    "Myocardial infarction": [
        [("MAP","<=",65), ("HR",">=",50), ("SpO2","<=",95), ("EtCO2","<=",35),
         ("CVP",">=",8), ("Urine","<=",30), ("Temp",">=",35.0), ("PI","<=",1.0)],
        [("MAP","<=",60), ("HR","<=",50), ("SpO2","<=",94), ("EtCO2","<=",32),
         ("CVP",">=",10), ("Urine","<=",20), ("Temp",">=",35.0), ("PI","<=",0.8)],
        [("MAP","<=",70), ("HR",">=",100), ("SpO2","<=",95), ("EtCO2","<=",35),
         ("CVP",">=",8), ("Urine","<=",25), ("Temp",">=",35.0), ("PI","<=",1.2)],
        [("MAP","<=",65), ("HR",">=",90), ("SpO2","<=",94), ("EtCO2","<=",33),
         ("CVP",">=",9), ("Urine","<=",20), ("Temp",">=",35.0), ("PI","<=",1.0)],
        [("MAP","<=",68), ("HR",">=",60), ("SpO2","<=",95), ("EtCO2","<=",34),
         ("CVP",">=",7), ("Urine","<=",30), ("Temp",">=",35.0), ("PI","<=",1.5)],
        [("MAP","<=",60), ("HR","<=",45), ("SpO2","<=",95), ("EtCO2","<=",32),
         ("CVP",">=",12), ("Urine","<=",20), ("Temp",">=",35.0), ("PI","<=",0.9)],
        [("MAP","<=",65), ("HR",">=",80), ("SpO2","<=",94), ("EtCO2","<=",33),
         ("CVP",">=",8), ("Urine","<=",25), ("Temp",">=",35.0), ("PI","<=",1.0)],
        [("MAP","<=",62), ("HR",">=",85), ("SpO2","<=",95), ("EtCO2","<=",32),
         ("CVP",">=",9), ("Urine","<=",18), ("Temp",">=",35.0), ("PI","<=",0.9)],
        [("MAP","<=",70), ("HR","<=",60), ("SpO2","<=",95), ("EtCO2","<=",35),
         ("CVP",">=",10), ("Urine","<=",22), ("Temp",">=",35.0), ("PI","<=",1.2)],
        [("MAP","<=",66), ("HR","<=",55), ("SpO2","<=",94), ("EtCO2","<=",33),
         ("CVP",">=",11), ("Urine","<=",20), ("Temp",">=",35.0), ("PI","<=",1.0)]
    ],

    # 9) Atrial fibrillation (proxy via HR instability / high HR)
    "Atrial fibrillation (proxy)": [
        [("HR",">=",110), ("PI","<=",1.0), ("MAP","<=",80), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",1), ("Urine","<=",30), ("Temp",">=",35.0)],
        [("HR",">=",120), ("PI","<=",1.2), ("MAP","<=",85), ("SpO2","<=",94),
         ("EtCO2","<=",40), ("CVP",">=",0), ("Urine","<=",20), ("Temp",">=",35.0)],
        [("HR",">=",100), ("PI","<=",0.9), ("MAP","<=",85), ("SpO2","<=",95),
         ("EtCO2","<=",42), ("CVP",">=",0), ("Urine","<=",25), ("Temp",">=",35.0)],
        [("HR",">=",110), ("PI","<=",0.8), ("MAP","<=",80), ("SpO2","<=",96),
         ("EtCO2","<=",40), ("CVP",">=",0), ("Urine","<=",30), ("Temp",">=",35.0)],
        [("HR",">=",115), ("PI","<=",1.0), ("MAP","<=",90), ("SpO2","<=",96),
         ("EtCO2","<=",42), ("CVP",">=",0), ("Urine","<=",30), ("Temp",">=",35.0)],
        [("HR",">=",105), ("PI","<=",0.9), ("MAP","<=",85), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",1), ("Urine","<=",20), ("Temp",">=",35.0)],
        [("HR",">=",120), ("PI","<=",0.7), ("MAP","<=",80), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",0), ("Urine","<=",15), ("Temp",">=",35.0)],
        [("HR",">=",100), ("PI","<=",1.1), ("MAP","<=",85), ("SpO2","<=",96),
         ("EtCO2","<=",42), ("CVP",">=",0), ("Urine","<=",25), ("Temp",">=",35.0)],
        [("HR",">=",110), ("PI","<=",0.8), ("MAP","<=",80), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",0), ("Urine","<=",10), ("Temp",">=",35.0)],
        [("HR",">=",115), ("PI","<=",0.9), ("MAP","<=",85), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",1), ("Urine","<=",30), ("Temp",">=",35.0)]
    ],

    # 10) AV blocks: 3rd degree (complete) - pattern: severe brady + low MAP/PI
    "Third-degree AV block": [
        [("HR","<=",40), ("MAP","<=",65), ("PI","<=",1.0), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",0), ("Urine","<=",30), ("Temp",">=",35.0)],
        [("HR","<=",35), ("MAP","<=",60), ("PI","<=",0.9), ("SpO2","<=",95),
         ("EtCO2","<=",38), ("CVP",">=",0), ("Urine","<=",20), ("Temp",">=",35.0)],
        [("HR","<=",45), ("MAP","<=",70), ("PI","<=",1.2), ("SpO2","<=",96),
         ("EtCO2","<=",40), ("CVP",">=",5), ("Urine","<=",25), ("Temp",">=",35.0)],
        [("HR","<=",40), ("MAP","<=",65), ("PI","<=",1.0), ("SpO2","<=",94),
         ("EtCO2","<=",40), ("CVP",">=",2), ("Urine","<=",15), ("Temp",">=",35.0)],
        [("HR","<=",38), ("MAP","<=",60), ("PI","<=",0.8), ("SpO2","<=",95),
         ("EtCO2","<=",38), ("CVP",">=",1), ("Urine","<=",10), ("Temp",">=",35.0)],
        [("HR","<=",42), ("MAP","<=",68), ("PI","<=",1.0), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",0), ("Urine","<=",20), ("Temp",">=",35.0)],
        [("HR","<=",40), ("MAP","<=",66), ("PI","<=",1.1), ("SpO2","<=",95),
         ("EtCO2","<=",38), ("CVP",">=",0), ("Urine","<=",15), ("Temp",">=",35.0)],
        [("HR","<=",35), ("MAP","<=",62), ("PI","<=",0.9), ("SpO2","<=",94),
         ("EtCO2","<=",37), ("CVP",">=",0), ("Urine","<=",12), ("Temp",">=",35.0)],
        [("HR","<=",40), ("MAP","<=",70), ("PI","<=",1.2), ("SpO2","<=",95),
         ("EtCO2","<=",40), ("CVP",">=",3), ("Urine","<=",25), ("Temp",">=",35.0)],
        [("HR","<=",39), ("MAP","<=",65), ("PI","<=",1.0), ("SpO2","<=",95),
         ("EtCO2","<=",39), ("CVP",">=",1), ("Urine","<=",18), ("Temp",">=",35.0)]
    ],

    # Second-degree AV block (proxy)
    "Second-degree AV block": [
        [("HR","<=",55), ("MAP","<=",75), ("PI","<=",1.5), ("SpO2","<=",96),
         ("EtCO2","<=",45), ("CVP",">=",0), ("Urine","<=",30), ("Temp",">=",35.0)],
        [("HR","<=",60), ("MAP","<=",78), ("PI","<=",1.6), ("SpO2","<=",96),
         ("EtCO2","<=",44), ("CVP",">=",0), ("Urine","<=",30), ("Temp",">=",35.0)],
        [("HR","<=",50), ("MAP","<=",70), ("PI","<=",1.3), ("SpO2","<=",95),
         ("EtCO2","<=",42), ("CVP",">=",0), ("Urine","<=",20), ("Temp",">=",35.0)],
        [("HR","<=",58), ("MAP","<=",75), ("PI","<=",1.8), ("SpO2","<=",96),
         ("EtCO2","<=",45), ("CVP",">=",0), ("Urine","<=",25), ("Temp",">=",35.0)],
        [("HR","<=",55), ("MAP","<=",72), ("PI","<=",1.5), ("SpO2","<=",95),
         ("EtCO2","<=",43), ("CVP",">=",0), ("Urine","<=",20), ("Temp",">=",35.0)],
        [("HR","<=",60), ("MAP","<=",80), ("PI","<=",2.0), ("SpO2","<=",96),
         ("EtCO2","<=",45), ("CVP",">=",0), ("Urine","<=",30), ("Temp",">=",35.0)],
        [("HR","<=",52), ("MAP","<=",76), ("PI","<=",1.4), ("SpO2","<=",95),
         ("EtCO2","<=",42), ("CVP",">=",0), ("Urine","<=",22), ("Temp",">=",35.0)],
        [("HR","<=",56), ("MAP","<=",74), ("PI","<=",1.6), ("SpO2","<=",95),
         ("EtCO2","<=",44), ("CVP",">=",0), ("Urine","<=",18), ("Temp",">=",35.0)],
        [("HR","<=",55), ("MAP","<=",70), ("PI","<=",1.2), ("SpO2","<=",95),
         ("EtCO2","<=",42), ("CVP",">=",0), ("Urine","<=",20), ("Temp",">=",35.0)],
        [("HR","<=",50), ("MAP","<=",72), ("PI","<=",1.3), ("SpO2","<=",95),
         ("EtCO2","<=",43), ("CVP",">=",0), ("Urine","<=",15), ("Temp",">=",35.0)]
    ],

    # First-degree AV block (proxy)
    "First-degree AV block": [
        [("HR",">=",45), ("HR","<=",90), ("MAP",">=",60), ("MAP","<=",95),
         ("SpO2",">=",90), ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",15)],
        [("HR",">=",40), ("HR","<=",95), ("MAP",">=",60), ("MAP","<=",100),
         ("SpO2",">=",90), ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",10)],
        [("HR",">=",45), ("HR","<=",85), ("MAP",">=",60), ("MAP","<=",95),
         ("SpO2",">=",90), ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",5)],
        [("HR",">=",48), ("HR","<=",92), ("MAP",">=",62), ("MAP","<=",98),
         ("SpO2",">=",91), ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",10)],
        [("HR",">=",50), ("HR","<=",90), ("MAP",">=",60), ("MAP","<=",95),
         ("SpO2",">=",92), ("EtCO2",">=",31), ("CVP",">=",0), ("Urine",">=",12)],
        [("HR",">=",45), ("HR","<=",88), ("MAP",">=",60), ("MAP","<=",95),
         ("SpO2",">=",90), ("EtCO2",">=",30), ("CVP",">=",1), ("Urine",">=",10)],
        [("HR",">=",46), ("HR","<=",91), ("MAP",">=",61), ("MAP","<=",96),
         ("SpO2",">=",90), ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",8)],
        [("HR",">=",45), ("HR","<=",89), ("MAP",">=",60), ("MAP","<=",95),
         ("SpO2",">=",90), ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",12)],
        [("HR",">=",47), ("HR","<=",93), ("MAP",">=",60), ("MAP","<=",95),
         ("SpO2",">=",91), ("EtCO2",">=",30), ("CVP",">=",0), ("Urine",">=",10)],
        [("HR",">=",45), ("HR","<=",92), ("MAP",">=",60), ("MAP","<=",95),
         ("SpO2",">=",92), ("EtCO2",">=",31), ("CVP",">=",0), ("Urine",">=",15)]
    ]
}

# ---- main evaluation function ----
def evaluate_reading(reading: Dict[str, float]) -> Dict[str, Any]:
    """
    :param reading: dict with keys SpO2, MAP, HR, EtCO2, CVP, Urine, Temp, PI
    :return: dict with 'diagnosis' (one of RULES keys), 'matched_variant' (index), 'explanation', 'conflicts' (list)
    """
    matched = []
    matched_details = {}

    for comp, variants in RULES.items():
        for idx, variant in enumerate(variants):
            if eval_variant(reading, variant):
                matched.append(comp)
                matched_details.setdefault(comp, []).append(idx+1)
                # break out to next complication if first variant matched? we gather all matches
                # (we want to find all possible matches, then resolve conflicts by priority)
                # but we do not break so that multiple variants for same comp are recorded.

    # If nothing matched -> return "No match" with nearest suggestions
    if not matched:
        # heuristics: if most sensors in normal ranges -> Normal
        # fallback: return Normal if any Normal variant matches
        if any(eval_variant(reading, v) for v in RULES["Normal"]):
            return {"diagnosis": "Normal", "matched_variant": RULES["Normal"], "explanation": "All vital signs in normal ranges", "conflicts": []}
        else:
            return {"diagnosis": "No rule matched", "matched_variant": [], "explanation": "No rule variant satisfied. Consider more detailed analysis or clinician review.", "conflicts": []}

    # If single matched, return it
    if len(set(matched)) == 1:
        comp = matched[0]
        idxs = matched_details.get(comp, [])
        # Build simple explanation: list top contributing sensors (those outside nominal)
        explanation = _build_explanation(reading, comp)
        return {"diagnosis": comp, "explanation": explanation, "conflicts": []}

    # If multiple matched -> resolve by PRIORITY
    unique = []
    for m in matched:
        if m not in unique:
            unique.append(m)

    # check priority order
    for p in PRIORITY:
        if p in unique:
            chosen = p
            break
    else:
        chosen = unique[0]

    conflicts = [c for c in unique if c != chosen]
    explanation = _build_explanation(reading, chosen)
    return {"diagnosis": chosen,
            "explanation": explanation, "conflicts": conflicts}


def _build_explanation(reading: Dict[str, float], comp_name: str) -> str:
    """
    Build a concise human-readable explanation: top 3 sensors that are most 'abnormal'
    relative to common intraop norms.
    (Simple heuristic: compute absolute z-like deviation vs typical baseline ranges)
    """
    # define nominal midpoints / ranges for scoring
    normals = {
        "SpO2": (97.0, 4.0),
        "MAP": (80.0, 12.0),
        "HR": (75.0, 20.0),
        "EtCO2": (38.0, 6.0),
        "CVP": (6.0, 3.0),
        "Urine": (50.0, 25.0),
        "Temp": (36.6, 0.7),
        "PI": (1.5, 0.8)
    }
    scores = []
    for k, (mid, sd) in normals.items():
        if k not in reading: continue
        dev = abs(reading[k] - mid) / (sd if sd>0 else 1.0)
        scores.append((dev, k, reading[k]))
    scores.sort(reverse=True)
    top = scores[:3]
    f = {}
    for (dev, k, val) in top:
        f[k] = [val, dev]
    return f

# ---- Example usage ----
#if __name__ == "__main__":
    # sample reading dictionary (simulate)
#sample = {"SpO2":99.0, "MAP":58.0, "HR":30.0, "EtCO2":38.0, "CVP":93.0, "Urine":8.0, "Temp":36.0, "PI":0.7}
#result = evaluate_reading(sample)
#print("Result:", result)

