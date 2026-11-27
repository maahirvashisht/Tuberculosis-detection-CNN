
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from model import train, predict_logits, sigmoid

BASE = Path(__file__).resolve().parent
NAMES = {"full": "CNN with dropout", "ablated": "CNN without dropout"}

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def confusion(y, scores, threshold):
    flags = scores >= threshold
    return tuple(int(a.sum()) for a in ((y==1)&flags,(y==1)&~flags,(y==0)&~flags,(y==0)&flags))

def choose_threshold(y, scores):
    options = []
    for t in np.unique(scores):
        tp,fn,tn,fp = confusion(y,scores,t)
        if tp/(tp+fn) >= .90:
            options.append((tn/(tn+fp),tp/(tp+fn),float(t)))
    return max(options)[2]

def wilson(k, n):
    z = 1.959964
    p = k/n
    d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0,center-half), min(1,center+half)

def performance(y, s, t):
    tp,fn,tn,fp = confusion(y,s,t)
    slo,shi = wilson(tp,tp+fn)
    clo,chi = wilson(tn,tn+fp)
    margins = s[y==1,None] - s[y==0]
    return dict(n=len(y),TP=tp,FN=fn,TN=tn,FP=fp,sensitivity=tp/(tp+fn),
                specificity=tn/(tn+fp),accuracy=(tp+tn)/len(y),sens_low=slo,sens_high=shi,
                spec_low=clo,spec_high=chi,auc=float(np.mean((margins>0)+.5*(margins==0))))

def paired_analysis(y, full, ablated):
    rng = np.random.default_rng(42)
    out = []
    for label,name in [(1,"sensitivity"),(0,"specificity")]:
        index = y==label
        delta = (full[index]==label).astype(int) - (ablated[index]==label).astype(int)
        boot = rng.choice(delta, (5000,len(delta)), replace=True).mean(axis=1)
        lo,hi = np.quantile(boot,[.025,.975])
        out.append(dict(metric=name,delta=float(delta.mean()),low=float(lo),high=float(hi)))
    b = int(((full==y)&(ablated!=y)).sum())
    c = int(((full!=y)&(ablated==y)).sum())
    m = b+c
    p = min(1.,2*sum(math.comb(m,k) for k in range(min(b,c)+1))/2**m)
    return {"paired":out,"full_only_correct":b,"ablated_only_correct":c,"mcnemar_p":p}

def main(cache, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = read_rows(BASE/"data/manifest.csv")
    with np.load(cache, allow_pickle=False) as data:
        if list(data["cases"]) != [r["case"] for r in records]:
            raise ValueError("The image cache does not match the supplied case manifest.")
        images = data["images"]
    y = np.array([int(r["y"]) for r in records], dtype=np.float32)
    splits = np.array([r["split"] for r in records])
    train_ids, val_ids, ext_ids = [np.flatnonzero(splits==s) for s in ("train","validation","external")]
    if tuple(map(len,(train_ids,val_ids,ext_ids))) != (496,166,138):
        raise ValueError("Unexpected cohort sizes.")
    variants = {}
    history = []
    for key,rate in [("full",.35),("ablated",0.)]:
        params,hist = train(images[train_ids],y[train_ids],images[val_ids],y[val_ids],rate)
        valid = sigmoid(predict_logits(images[val_ids],params))
        threshold = choose_threshold(y[val_ids], valid)
        epoch = min(hist,key=lambda r:r["val_loss"])["epoch"]
        variants[key] = (params,valid,threshold,epoch)
        np.savez_compressed(output/(key+".npz"),**params)
        history.extend(dict(model=NAMES[key],**r) for r in hist)
    metrics, prediction_rows, flags = [], [], {}
    for key,(params,valid,threshold,epoch) in variants.items():
        ext = sigmoid(predict_logits(images[ext_ids],params))
        flags[key] = ext >= threshold
        for split,ids,scores in [("validation",val_ids,valid),("external",ext_ids,ext)]:
            for rule,t in [("selected",threshold),("default",.5)]:
                metrics.append(dict(model=NAMES[key],split=split,rule=rule,threshold=t,**performance(y[ids],scores,t)))
            for i,s in zip(ids,scores):
                prediction_rows.append(dict(case=records[i]["case"],model=NAMES[key],split=split,
                    y=int(y[i]),score=float(s),cutoff=threshold,predicted=int(s>=threshold)))
    baseline = []
    for r in read_rows(BASE/"data/metrics.csv"):
        baseline.append({k:(r[k] if k in ("split","model","rule") else int(r[k]) if k in ("n","TP","FN","TN","FP") else float(r[k])) for k in r})
    summary = dict(threshold=variants["full"][2],seed=42,target=.90,epochs=20,
                   best_epoch={k:v[3] for k,v in variants.items()},metrics=metrics,baseline=baseline,
                   **paired_analysis(y[ext_ids],flags["full"],flags["ablated"]))
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    write_rows(output/"cnn_metrics.csv",metrics)
    write_rows(output/"predictions.csv",prediction_rows)
    write_rows(output/"history.csv",history)
    write_rows(output/"paired_differences.csv",summary["paired"])
    print("Saved results to",output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("--out",default=str(BASE/"results"))
    args=parser.parse_args()
    main(args.cache,args.out)
