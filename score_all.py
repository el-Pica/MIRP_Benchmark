import json, os, glob

base = r"C:\Users\xPica\Documents\NII Research\MIRP_Benchmark\results"

def score_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    correct = total = 0
    for entry in data:
        for r in entry["results_call"]:
            ans = r["model_answer"].strip()[:1]
            exp = str(r["expected_answer"]).strip()[:1]
            total += 1
            if ans == exp:
                correct += 1
    return correct, total

groups = {}
patterns = ["RQ3_*.json", "AS_*.json", "RQ2_*.json", "qa_all_*.json"]
for pat in patterns:
    for fp in sorted(glob.glob(os.path.join(base, pat))):
        name = os.path.basename(fp)
        key = name.rsplit("_add_run_", 1)[0]
        groups.setdefault(key, []).append(fp)

print(f"{'Key':<55} {'Run0':>6} {'Run1':>6} {'Run2':>6} {'Mean':>7}")
print("-" * 82)
for key in sorted(groups):
    files = sorted(groups[key])
    accs = []
    for fp in files:
        c, t = score_file(fp)
        accs.append(c/t if t else 0)
    mean = sum(accs)/len(accs)
    parts = [f"{a:.3f}" for a in accs]
    while len(parts) < 3:
        parts.append("     -")
    print(f"{key:<55} {parts[0]:>6} {parts[1]:>6} {parts[2]:>6} {mean:>7.3f}")
