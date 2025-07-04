#!/usr/bin/env python3
"""
augment_ply_color.py  –  Data-augmentation de nubes .ply con etiquetas por color.

Salida: 4 copias por nube (jitter, rot, trans, scale) en TXT o PLY.

USO EJEMPLO
-----------
python augment_ply_color.py data_ply data_aug \
        --out_fmt txt --jitter_sigma 0.02 --rot_axis xyz --seed 42
"""

import os, argparse, random, math, numpy as np
import open3d as o3d               # pip install open3d

# ------------------------------------------------------------------------
# Configuración y argumentos
# ------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Augmentación xyz de .ply con color-label; salida en txt o ply.")
    p.add_argument("input_dir",  help="Carpeta raíz con .ply de entrada")
    p.add_argument("output_dir", help="Carpeta donde se escriben los archivos aumentados")
    p.add_argument("--out_fmt",  choices=["txt", "ply"], default="txt",
                   help="Formato de salida (txt = x y z label)  [txt]")
    # Augment params
    p.add_argument("--jitter_sigma", type=float, default=0.01,
                   help="σ del ruido gaussiano (jitter)          [0.01]")
    p.add_argument("--rot_axis", choices=["z", "xyz"], default="z",
                   help="Eje de rotación aleatoria (z|xyz)        [z]")
    p.add_argument("--rot_max_deg", type=float, default=180,
                   help="Ángulo máx. de rotación (°)              [180]")
    p.add_argument("--trans_max", type=float, default=0.5,
                   help="Magnitud máx. de traslación              [0.5]")
    p.add_argument("--scale_min", type=float, default=0.8)
    p.add_argument("--scale_max", type=float, default=1.25)
    # Otros
    p.add_argument("--seed", type=int, default=None,
                   help="Semilla para reproducibilidad")
    return p.parse_args()

# ------------------------------------------------------------------------
#  Colo-label helpers
# ------------------------------------------------------------------------

# Colores ‘puros’ (0-255) → etiqueta
COLOR_TO_LBL = {
    (  0,   0, 255): 0,   # Azul
    (255,   0,   0): 1,   # Rojo
    (  0, 255,   0): 2    # Verde
}
# Etiqueta → color (para volver a PLY)
LBL_TO_COLOR = {v: k for k, v in COLOR_TO_LBL.items()}

def color_to_label(rgb, atol=50):
    """Convierte un color (0-255) a la etiqueta 0/1/2/3, con tolerancia ±atol."""
    for col, lbl in COLOR_TO_LBL.items():
        if np.allclose(rgb, col, atol=atol):
            return lbl
    return 3  # Otro

def labels_from_colors(colors):
    """array Nx3 (0-255 o 0-1) → Nx1 etiquetas."""
    if colors.max() <= 1.0:      # normalizado
        colors = (colors * 255).round().astype(int)
    else:
        colors = colors.astype(int)
    return np.array([color_to_label(c) for c in colors]).reshape(-1, 1)

def colors_from_labels(labels):
    """Nx1 etiquetas → Nx3 uint8 (0-255)."""
    out = np.zeros((labels.size, 3), dtype=np.uint8)
    for lbl, rgb in LBL_TO_COLOR.items():
        out[labels[:, 0] == lbl] = rgb
    return out

# ------------------------------------------------------------------------
#  Augmentations (xyz only)
# ------------------------------------------------------------------------

def jitter(xyz, sigma, rng):            return xyz + rng.normal(0, sigma, xyz.shape)

def rot_mat(rng, axis="z", max_deg=180):
    if axis == "z":
        ang = math.radians(rng.uniform(-max_deg, max_deg))
        c, s = math.cos(ang), math.sin(ang)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    # 3D random axis
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    ang = math.radians(rng.uniform(-max_deg, max_deg))
    c, s = math.cos(ang), math.sin(ang); x, y, z = v
    R = np.array([
        [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)] ])
    return R

def rotate(xyz, R):                     return xyz @ R.T
def translate(xyz, max_shift, rng):     return xyz + rng.uniform(-max_shift, max_shift, 3)
def scale(xyz, smin, smax, rng):        return xyz * rng.uniform(smin, smax)

# ------------------------------------------------------------------------
#  IO helpers
# ------------------------------------------------------------------------

def read_ply_xyz_label(path):
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points, dtype=np.float32)
    colors = np.asarray(pc.colors, dtype=np.float32)
    lbl  = labels_from_colors(colors)
    return xyz, lbl

def write_txt(path, xyz, lbl):
    data = np.hstack([xyz, lbl])
    np.savetxt(path, data, fmt="%.6f")

def write_ply(path, xyz, lbl):
    pc = o3d.geometry.PointCloud()
    pc.points  = o3d.utility.Vector3dVector(xyz)
    pc.colors  = o3d.utility.Vector3dVector(colors_from_labels(lbl) / 255.0)  # normalizar a 0-1
    o3d.io.write_point_cloud(path, pc, write_ascii=True)

# ------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------

def augment_one(xyz, lbl, a, rng, out_base, write_fn):
    # Jitter
    write_fn(f"{out_base}_jit.{a}", jitter(xyz, a.jitter_sigma, rng), lbl)
    # Rot
    R = rot_mat(rng, a.rot_axis, a.rot_max_deg); write_fn(f"{out_base}_rot.{a}", rotate(xyz, R), lbl)
    # Trans
    write_fn(f"{out_base}_tra.{a}", translate(xyz, a.trans_max, rng), lbl)
    # Scale
    write_fn(f"{out_base}_scl.{a}", scale(xyz, a.scale_min, a.scale_max, rng), lbl)

def main():
    a = get_args()
    rng = np.random.default_rng(a.seed)
    ext = "txt" if a.out_fmt == "txt" else "ply"
    write_fn = write_txt if a.out_fmt == "txt" else write_ply

    for root, _, files in os.walk(a.input_dir):
        for fn in files:
            if not fn.lower().endswith(".ply"):  # sólo PLY
                continue
            in_path = os.path.join(root, fn)
            rel_dir = os.path.relpath(root, a.input_dir)
            out_dir = os.path.join(a.output_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(fn)[0]
            xyz, lbl = read_ply_xyz_label(in_path)
            base = os.path.join(out_dir, stem)
            augment_one(xyz, lbl, a, rng, base, lambda p, X, L: write_fn(f"{p}.{ext}", X, L))
            print(f"✅  {in_path}  →  4 copias en {out_dir}")

if __name__ == "__main__":
    main()
