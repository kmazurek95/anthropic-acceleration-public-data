import sys, zipfile, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[3])
import pandas as pd
from eci import prepare_benchmark_data, fit_eci_model
from accel import inputs
A, B = sys.argv[1], sys.argv[2]
da, db = prepare_benchmark_data(A), prepare_benchmark_data(B)
print("prepared shapes:", da.shape, db.shape, "models", da.model.nunique(), db.model.nunique(), "benchmarks", da.benchmark.nunique(), db.benchmark.nunique())
key = list(da.columns)
sa = da.sort_values(key).reset_index(drop=True); sb = db.sort_values(key).reset_index(drop=True)
print("prepared data identical:", sa.equals(sb))
if not sa.equals(sb):
    m = sa.merge(sb, how="outer", indicator=True); print(m[m._merge != "both"].to_string())
la, _, _ = fit_eci_model(da, bootstrap_samples=0); lb, _, _ = fit_eci_model(db, bootstrap_samples=0)
j = la[["Model","eci"]].merge(lb[["Model","eci"]], on="Model", how="outer", suffixes=("_d","_a"))
print("LS eci max abs diff:", (j.eci_d - j.eci_a).abs().max(), "n:", len(j))
models = sorted(da.model.unique())
def tab(z):
    with zipfile.ZipFile(z) as f: pub = pd.read_csv(f.open("epoch_capabilities_index/eci_scores.csv"))
    t, _ = inputs.group_table(inputs.read_model_metadata(z), models, org_fallback=dict(zip(pub.Model, pub.Organization)), allow_unknown_org=True)
    return t.sort_values("model").reset_index(drop=True)
ta, tb = tab(A), tab(B)
print("group table (date/org/accessibility for fitted groups) identical:", ta.equals(tb))
if not ta.equals(tb):
    m = ta.merge(tb, how="outer", indicator=True); print(m[m._merge != "both"].to_string())
# What changed in model_metadata overall
ma, mb = inputs.read_model_metadata(A), inputs.read_model_metadata(B)
print("model_metadata rows direct/archived:", len(ma), len(mb))
m = ma.astype(str).merge(mb.astype(str), how="outer", indicator=True)
d = m[m._merge != "both"]
print("differing metadata rows:", len(d), "| groups:", sorted(d.model_group.unique()))
print("of which fitted groups:", sorted(set(d.model_group) & set(models)))
for n in ["mirrorcode.csv","spireclimb.csv","spireclimb_sts1_a20_heart.csv","bool_q_external.csv","gsm8k_external.csv","hella_swag_external.csv"]:
    with zipfile.ZipFile(A) as fa, zipfile.ZipFile(B) as fb:
        xa, xb = fa.read(n).decode().splitlines(), fb.read(n).decode().splitlines()
    ch = [(i, a, b) for i, (a, b) in enumerate(zip(xa, xb)) if a != b]
    print(f"{n}: lines {len(xa)}/{len(xb)}, changed lines {len(ch)}; first:", ch[:1])
