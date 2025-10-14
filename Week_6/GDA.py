# xml_to_predictions.py
# 只用 O-A0038-003.xml 作為輸入，輸出整張網格的 C、R、h 到 predictions.xlsx

from __future__ import annotations
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd

FILENAME = "O-A0038-003.xml"
OUT_XLSX = "predictions.xlsx"

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

    temp = grid
    C = (temp > -900.0).astype(int)  # 有效遮罩
    R = temp.copy()                  # 在格點上，R(x) 直接用原始值
    h = np.where(C == 1, R, -999.0)  # 你的定義

    # 展平成表格
    df = pd.DataFrame({
        "lon": lon_grid.ravel(),
        "lat": lat_grid.ravel(),
        "C":   C.ravel(),
        "R":   R.ravel(),
        "h":   h.ravel(),
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
