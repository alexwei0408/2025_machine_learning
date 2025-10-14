# xml_to_predictions.py
# 只用 O-A0038-003.xml 作為輸入，輸出整張網格的 C、R、h 到 predictions.xlsx

from __future__ import annotations
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd

FILENAME = "O-A0038-003.xml"
OUT_XLSX = "predictions.xlsx"

# --- add these near the top (after imports) ---
def inside(lon, lat, meta):
    return (meta["blon"] <= lon <= meta["blon"]+(meta["nx"]-1)*meta["dlon"] and
            meta["blat"] <= lat <= meta["blat"]+(meta["ny"]-1)*meta["dlat"])

def nearest_index(lon, lat, meta):
    ix = int((lon-meta["blon"])/meta["dlon"] + 0.5)
    iy = int((lat-meta["blat"])/meta["dlat"] + 0.5)
    ix = max(0, min(meta["nx"]-1, ix)); iy = max(0, min(meta["ny"]-1, iy))
    return ix, iy

def C_of(lon, lat, grid, meta):
    if not inside(lon, lat, meta): return 0
    ix, iy = nearest_index(lon, lat, meta)
    return 1 if grid[iy, ix] > -900.0 else 0  # -999 means invalid

class Regressor:
    """R(x): bilinear interpolation; fallback to locally weighted linear regression (LWLR)."""
    def __init__(self, grid, meta, tau=0.08, k=80):
        self.G, self.m, self.tau, self.k = grid, meta, tau, k
        val = (grid > -900.0)
        ys, xs = np.where(val)
        lons = meta["blon"] + xs*meta["dlon"]
        lats = meta["blat"] + ys*meta["dlat"]
        self.Xv = np.c_[lons, lats]; self.yv = grid[val]

    def _bilinear(self, lon, lat):
        fx = (lon-self.m["blon"])/self.m["dlon"]; fy = (lat-self.m["blat"])/self.m["dlat"]
        ix = int(np.floor(fx)); iy = int(np.floor(fy))
        ix = max(0, min(self.m["nx"]-2, ix)); iy = max(0, min(self.m["ny"]-2, iy))
        val = (self.G > -900.0)
        if not (val[iy,ix] and val[iy,ix+1] and val[iy+1,ix] and val[iy+1,ix+1]): return None
        x1 = self.m["blon"] + ix*self.m["dlon"]; x2 = x1 + self.m["dlon"]
        y1 = self.m["blat"] + iy*self.m["dlat"]; y2 = y1 + self.m["dlat"]
        tx = (lon-x1)/(x2-x1); ty = (lat-y1)/(y2-y1)
        Q11, Q21 = self.G[iy,ix],   self.G[iy,ix+1]
        Q12, Q22 = self.G[iy+1,ix], self.G[iy+1,ix+1]
        return Q11*(1-tx)*(1-ty) + Q21*tx*(1-ty) + Q12*(1-tx)*ty + Q22*tx*ty

    def _lwlr(self, lon, lat):
        d2 = (self.Xv[:,0]-lon)**2 + (self.Xv[:,1]-lat)**2
        idx = np.argsort(d2)[: self.k]
        Xk, yk = self.Xv[idx], self.yv[idx]
        Phi = np.c_[np.ones(len(Xk)), Xk]
        w = np.exp(-d2[idx]/(2*self.tau**2))
        A = Phi.T @ (w[:,None]*Phi) + 1e-8*np.eye(3)
        b = Phi.T @ (w*yk)
        theta = np.linalg.solve(A, b)
        return float(np.array([1.0, lon, lat]) @ theta)

    def predict(self, lon, lat):
        if not inside(lon, lat, self.m): return np.nan
        v = self._bilinear(lon, lat)
        return v if v is not None else self._lwlr(lon, lat)

def h_of(lon, lat, grid, meta, reg):
    """piecewise: R(x) if C(x)=1, else -999"""
    return reg.predict(lon, lat) if C_of(lon, lat, grid, meta)==1 else -999.0

def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def resolve_xml_path(filename: str = FILENAME) -> Path:
    here = script_dir()
    cand = [
        here / filename,
        Path.cwd() / filename,
        Path.home() / "Desktop" / filename,
        Path.home() / "Downloads" / filename,
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {filename}，請將檔案放在腳本同資料夾或提供正確路徑。")

def load_grid_from_xml(xml_path: Path):
    ns = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}
    root = ET.parse(str(xml_path)).getroot()

    blon = float(root.findtext(".//cwa:BottomLeftLongitude", namespaces=ns))
    blat = float(root.findtext(".//cwa:BottomLeftLatitude",  namespaces=ns))
    tlon = float(root.findtext(".//cwa:TopRightLongitude",   namespaces=ns))
    tlat = float(root.findtext(".//cwa:TopRightLatitude",    namespaces=ns))

    # 規格：67 × 120，解析度 0.03°
    nx, ny = 67, 120
    dlon = dlat = 0.03

    content = root.findtext(".//cwa:Content", namespaces=ns)
    vals = [float(t.strip()) for t in content.replace("\n", ",").split(",") if t.strip()]
    grid = np.array(vals, dtype=float).reshape(ny, nx)

    meta = dict(blon=blon, blat=blat, tlon=tlon, tlat=tlat, nx=nx, ny=ny, dlon=dlon, dlat=dlat)
    return grid, meta

def export_full_predictions(xml_path: Path, out_xlsx: Path) -> Path:
    grid, meta = load_grid_from_xml(xml_path)
    blon, blat = meta["blon"], meta["blat"]
    nx, ny, dlon, dlat = meta["nx"], meta["ny"], meta["dlon"], meta["dlat"]

   # 建立 lon/lat 網格
    lons = blon + np.arange(nx) * dlon
    lats = blat + np.arange(ny) * dlat
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    C = (grid > -900.0).astype(int)   # classifier on nodes
    reg = Regressor(grid, meta, tau=0.08, k=80)

    # 對每個格點估計 R_pred（用我們的回歸器），再依規則組合 h
    R_pred = np.array([reg.predict(lon, lat) for lon, lat in np.c_[lon_grid.ravel(), lat_grid.ravel()]]).reshape(grid.shape)
    H = np.where(C==1, R_pred, -999.0)

    # 也一併輸出 ground truth（在節點上就是原始網格值）
    df = pd.DataFrame({
        "lon": lon_grid.ravel(),
        "lat": lat_grid.ravel(),
        "C":   C.ravel(),
        "R_true": grid.ravel(),
        "R_pred": R_pred.ravel(),
        "h":   H.ravel(),
    })


    out_xlsx = Path(out_xlsx)
    df.to_excel(out_xlsx, index=False)
    return out_xlsx

def main():
    xml_path = resolve_xml_path(FILENAME)
    out = export_full_predictions(xml_path, script_dir() / OUT_XLSX)
    print(f"[Saved] {out}")

if __name__ == "__main__":
    main()
