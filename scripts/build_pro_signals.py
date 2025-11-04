#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRO sygnały:
- DMACD (8,17,9) + dywergencje
- 3x3 (SMA3 shift 3) diagnostyka thrust
- Fibo retrace (0.382/0.618) + ABC extensions (COP/OP/XOP)
- VWAP dzienny (fallback do TWAP gdy brak V)
- POC z profilu (Volume-mid gdy jest volume, inaczej TPO)
- Confluence: klastrowanie poziomów (Fibo, EMA20/50/200, Swing H/L, VWAP, POC, Round)
Wejście: data/<PAIR>.csv
Wyjście: signals/pro.json + pomocnicze signals/levels_*.csv, signals/poc_*.csv
"""

import os, json, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

DATA_DIR="data"; OUT_DIR="signals"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

DMACD=(8,17,9)
PROFILE_LOOKBACK_H=72
PROFILE_BIN_MULT=4
CONF_TOL_ATR=0.15
ROUND_STEP={"BTCUSD":100.0,"XAUUSD":1.0,"DEFAULT":0.005}

def read_df(pair):
    p=Path(DATA_DIR)/f"{pair}.csv"
    if not p.exists(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    tcol=next((c for c in ["time","timestamp","datetime","date","Date","datetime_utc"] if c in df.columns),None)
    if not tcol: raise KeyError(f"{p}: brak kolumny czasu")
    df["time"]=pd.to_datetime(df[tcol],utc=True,errors="coerce")
    df=df.rename(columns={c:c.lower() for c in df.columns})
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        else: df[c]=np.nan
    now_full=datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0)
    df=df[df["time"]<now_full].dropna(subset=["open","high","low","close"])
    df=df.sort_values("time").drop_duplicates("time",keep="last").reset_index(drop=True)
    return df

def ema(s,span): return s.ewm(span=span,adjust=False).mean()
def sma(s,n): return s.rolling(n,min_periods=n).mean()
def atr(df,n=14):
    pc=df["close"].shift(1)
    tr=pd.concat([(df["high"]-df["low"]).abs(),(df["high"]-pc).abs(),(df["low"]-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def dmacd(close,fast,slow,sig):
    line=ema(close,fast)-ema(close,slow); signal=ema(line,sig); hist=line-signal
    return line,signal,hist
def dma33(close): return sma(close,3).shift(3)
def thrust_flags(close,dma,above=True):
    cond=(close>dma) if above else (close<dma); run=[];cnt=0
    for ok in cond.fillna(False): cnt=cnt+1 if ok else 0; run.append(cnt)
    return pd.Series(run,index=close.index)
def pip_size(pair):
    if pair=="BTCUSD": return 1.0
    if pair=="XAUUSD": return 0.01
    if pair.endswith("JPY"): return 0.01
    return 0.0001

def zigzag(df,rev_pct=0.003):
    c=df["close"].to_numpy()
    if len(c)<10: return []
    piv=[(0,c[0])]; last_i=0; last_p=c[0]
    for i in range(1,len(c)):
        if abs(c[i]-last_p)>=rev_pct*max(1e-9,last_p):
            piv.append((i,c[i])); last_i=i; last_p=c[i]
    if piv[-1][0]!=len(c)-1: piv.append((len(c)-1,c[-1]))
    out=[]; seen=set()
    for i,p in piv:
        if i not in seen: out.append((i,p)); seen.add(i)
    return out

def fibo_retrace(a,b):
    hi,lo=(b,a) if b>a else (a,b); rng=hi-lo
    r382=hi-0.382*rng if b>a else lo+0.382*rng
    r618=hi-0.618*rng if b>a else lo+0.618*rng
    return {"R382":r382,"R618":r618}
def fibo_abc(a,b,c):
    ab=b-a
    return {"COP":c+0.618*ab,"OP":c+1.0*ab,"XOP":c+1.618*ab}

def session_vwap(df):
    tp=(df["high"]+df["low"]+df["close"])/3.0
    v=df["volume"].fillna(0)
    if v.sum()<=0: v=pd.Series(1.0,index=df.index)  # TWAP fallback
    day=df["time"].dt.floor("D")
    num=(tp*v).groupby(day).cumsum(); den=v.groupby(day).cumsum()
    return num/den

def profile_poc(df,pair,hours=PROFILE_LOOKBACK_H):
    wstart=df["time"].iloc[-1]-pd.Timedelta(hours=hours)
    w=df[df["time"]>=wstart].copy()
    if w.empty: return None,"tpo"
    ps=pip_size(pair)*PROFILE_BIN_MULT
    lo=w["low"].min(); hi=w["high"].max()
    nbins=int(max(10,min(400,math.ceil((hi-lo)/max(ps,1e-9)))))
    edges=np.linspace(lo,hi,nbins+1); centers=(edges[:-1]+edges[1:])/2.0
    agg=np.zeros(nbins)
    vol_ok=w["volume"].fillna(0).sum()>0
    mids=((w["high"]+w["low"])/2.0).to_numpy()
    idx=np.clip(np.searchsorted(edges,mids)-1,0,nbins-1)
    if vol_ok:
        for i,v in zip(idx,w["volume"].fillna(0).to_numpy()): agg[i]+=float(v)
        used="vol_mid"
    else:
        for i in idx: agg[i]+=1.0
        used="tpo"
    poc_i=int(np.argmax(agg)); poc=float(centers[poc_i])
    pd.DataFrame({"price":centers,"profile":agg}).to_csv(Path(OUT_DIR)/f"poc_{pair}.csv",index=False)
    return poc,used

def round_level(pair,price):
    step=ROUND_STEP.get(pair,ROUND_STEP["DEFAULT"])
    return round(price/step)*step

def confluence(df,pair):
    levels=[]
    # Fibo
    zz=zigzag(df,rev_pct=0.005 if pair.endswith("JPY") else 0.003)
    if len(zz)>=3:
        a,b,c=zz[-3][1],zz[-2][1],zz[-1][1]
        retr=fibo_retrace(a,b); ext=fibo_abc(a,b,c)
        for k,v in retr.items(): levels.append((float(v),f"Fibo {k}"))
        for k,v in ext.items():  levels.append((float(v),f"Fibo {k}"))
        pd.DataFrame([{"pair":pair,**{**retr,**ext}}]).to_csv(Path(OUT_DIR)/f"levels_{pair}.csv",index=False)
    # EMA
    for name,val in [("EMA20",ema(df["close"],20).iloc[-1]),
                     ("EMA50",ema(df["close"],50).iloc[-1]),
                     ("EMA200",ema(df["close"],200).iloc[-1])]:
        if pd.notna(val): levels.append((float(val),name))
    # Swing H/L
    levels += [(float(df["high"].iloc[-20:].max()),"SwingH"),
               (float(df["low"].iloc[-20:].min()),"SwingL")]
    # VWAP d
    vwap = session_vwap(df).iloc[-1]
    if pd.notna(vwap): levels.append((float(vwap),"VWAPd"))
    # POC
    poc,used = profile_poc(df,pair)
    if poc is not None: levels.append((float(poc), f"POC[{used}]"))
    # Round
    levels.append((float(round_level(pair, df['close'].iloc[-1])), "Round"))

    # klastrowanie w strefy
    atr14=atr(df,14).iloc[-1]
    tol=max((CONF_TOL_ATR*float(atr14)) if pd.notna(atr14) else 0.0, pip_size(pair)*4)
    levels=sorted(levels,key=lambda x:x[0])
    clusters=[]; cur=[levels[0]]
    for p,l in levels[1:]:
        if abs(p-np.mean([x[0] for x in cur]))<=tol: cur.append((p,l))
        else: clusters.append(cur); cur=[(p,l)]
    if cur: clusters.append(cur)
    zones=[]
    for cl in clusters:
        prices=[x[0] for x in cl]; labels=[x[1] for x in cl]
        score=len(cl)+0.5*sum(lbl.startswith("Fibo") for lbl in labels)+0.5*sum(lbl.startswith("EMA") for lbl in labels)
        zones.append({"center":float(np.mean(prices)),"low":float(min(prices)),
                      "high":float(max(prices)),"labels":labels,"score":round(score,2),"tol":float(tol)})
    zones=sorted(zones,key=lambda z:(-z["score"], abs(df["close"].iloc[-1]-z["center"])))
    return zones

def dmacd_divs(df,lookback=80):
    line,sig,hist=dmacd(df["close"],*DMACD)
    out=[]
    w=df.iloc[-lookback:].copy(); h=hist.iloc[-lookback:]; c=w["close"]
    # minima
    cmin_i=int(c.idxmin()); prev=w[c.index < cmin_i]
    if len(prev)>=3:
        cmin2_i=int(prev["close"].idxmin())
        if c.loc[cmin_i] < c.loc[cmin2_i] and h.loc[cmin_i] > h.loc[cmin2_i]:
            out.append({"type":"div_bull_regular","last":str(df['time'].loc[cmin_i])})
        if c.loc[cmin_i] > c.loc[cmin2_i] and h.loc[cmin_i] < h.loc[cmin2_i]:
            out.append({"type":"div_bull_hidden","last":str(df['time'].loc[cmin_i])})
    # maxima
    cmax_i=int(c.idxmax()); prev=w[c.index < cmax_i]
    if len(prev)>=3:
        cmax2_i=int(prev["close"].idxmax())
        if c.loc[cmax_i] > c.loc[cmax2_i] and h.loc[cmax_i] < h.loc[cmax2_i]:
            out.append({"type":"div_bear_regular","last":str(df['time'].loc[cmax_i])})
        if c.loc[cmax_i] < c.loc[cmax2_i] and h.loc[cmax_i] > h.loc[cmax2_i]:
            out.append({"type":"div_bear_hidden","last":str(df['time'].loc[cmax_i])})
    return out, float(line.iloc[-1]) if pd.notna(line.iloc[-1]) else None, \
           float(sig.iloc[-1]) if pd.notna(sig.iloc[-1]) else None, \
           float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else None

def one_pair(pair):
    df=read_df(pair)
    divs,ml,ms,mh=dmacd_divs(df)
    zones=confluence(df,pair)
    dma=dma33(df["close"]); up=int(thrust_flags(df["close"],dma,True).iloc[-1] or 0)
    dn=int(thrust_flags(df["close"],dma,False).iloc[-1] or 0)
    return {"pair":pair,"last_time":str(df["time"].iloc[-1]),"close":float(df["close"].iloc[-1]),
            "macd":{"line":ml,"signal":ms,"hist":mh,"divergences":divs},
            "confluence":zones[:5],"thrust":{"up_bars":up,"down_bars":dn}}

def main():
    items=[]
    pairs=sorted([f.stem.upper() for f in Path(DATA_DIR).glob("*.csv")])
    if "PAIRS" in os.environ:
        wl=[s.strip().upper() for s in os.environ["PAIRS"].split(",") if s.strip()]
        pairs=[p for p in pairs if p in wl]
    for pair in pairs:
        try: items.append(one_pair(pair))
        except Exception as e: print(f"[{pair}] ERROR: {e}")
    out={"generated_at":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"items":items}
    with open(Path(OUT_DIR)/"pro.json","w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(f"Zapisano signals/pro.json (pairs: {len(items)})")

if __name__=="__main__":
    main()
