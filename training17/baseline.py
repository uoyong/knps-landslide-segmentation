import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import joblib
import shapely
import torch
import torch.nn.functional as F
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
CKPT = Path("14_submission/assets/model/best_decoder.pt")
SAM3_DIR = Path("14_submission/assets/model/sam3")

PATCH = 256       # 입력 패치 해상도 (px) — 좌표 규약의 기준 프레임
IMG = 1024        # 렌더 해상도 (4x 업샘플)
UP = IMG // PATCH
MODEL_IN = 1008   # SAM3 입력 해상도
M2_PER_PX2 = 100.0            # 1px = 10m
PROMPT_PAD = .15  # representative validation: best fixed box expansion
PRESENCE_MODEL = Path("14_submission/assets/presence.joblib")


# ---------- 입력 트리 ----------

def load_tree(input_dir: Path) -> dict:
    """입력 트리 -> {창ID: {"sites": [...], "dates": {날짜: npy 경로}}}.

    AIF_INPUT_DIR 에는 업로드한 input.zip 이 **풀린 트리**로 주입된다(플랫폼 공식
    동작이며 프로브 제출로 실측 확인함). 따라서 zip 해제 로직은 두지 않는다.
    rglob 이라 감싸는 폴더가 한 겹 끼어 있어도 찾아낸다.
    """
    locations = {}
    for sj in sorted(input_dir.rglob("sites.json")):
        loc = sj.parent.name
        dates = {npy.stem: npy for npy in sorted((sj.parent / "bands").glob("*.npy"))}
        if dates:
            locations[loc] = {"sites": json.loads(sj.read_text()), "dates": dates}
    if not locations:
        raise SystemExit(
            f"입력 트리(<창ID>/sites.json + <창ID>/bands/<날짜>.npy)를 찾지 못함: {input_dir}")
    return locations


def polygons_to_geom(polys):
    """3중 배열(MultiPolygon 좌표 규약) -> shapely geometry 또는 None."""
    parts = []
    for part in polys or []:
        if not part or len(part[0]) < 3:
            continue
        holes = [r for r in part[1:] if len(r) >= 3]
        try:
            p = Polygon(part[0], holes)
        except (ValueError, TypeError):
            continue
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty and p.area > 0:
            parts.append(p)
    if not parts:
        return None
    g = unary_union(parts)
    return g if (not g.is_empty and g.area > 0) else None


def geom_to_polys(g):
    """shapely geometry -> 3중 배열 (좌표 소수 3자리)."""
    if g is None or g.is_empty:
        return []
    parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
    out = []
    for p in parts:
        rings = [p.exterior, *p.interiors]
        out.append([[[round(x, 3), round(y, 3)] for x, y in r.coords] for r in rings])
    return out


# ---------- 입력 렌더링 ----------

def render_rgb(bands: np.ndarray) -> Image.Image:
    """B04/B03/B02 -> 공통 1~99% 스트레치 + 감마 0.7 -> 1024px RGB."""
    rgb = bands[:, :, :3].astype(np.float32)   # R,G,B
    lo, hi = np.percentile(rgb, (1, 99))
    u8 = (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.7 * 255).astype(np.uint8)
    return Image.fromarray(u8).resize((IMG, IMG), Image.LANCZOS)


def px_polys_from_mask(mask: np.ndarray, min_area_px2: float):
    """이진 마스크(1024) -> 256px 픽셀좌표 폴리곤 목록.

    행 단위 런(run)을 사각형으로 만들어 합집합한다 — 픽셀 경계를 그대로 따르는
    폴리곤이 나오므로 결과는 래스터 벡터화(GDAL polygonize)와 같다.
    """
    boxes = []
    for r in np.flatnonzero(mask.any(axis=1)):
        pad = np.concatenate(([0], mask[r].view(np.int8), [0]))
        edges = np.flatnonzero(np.diff(pad))
        for s, e in zip(edges[::2], edges[1::2]):
            boxes.append(box(s / UP, r / UP, e / UP, (r + 1) / UP))
    if not boxes:
        return []
    g = unary_union(boxes)
    parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
    return [p for p in parts if p.area >= min_area_px2]


# ---------- dNDVI 스크린 ----------

def rasterize(geom, shape) -> np.ndarray:
    """폴리곤 -> 픽셀중심(정수+0.5) 포함 여부 불리언 마스크."""
    out = np.zeros(shape, dtype=bool)
    if geom is None or geom.is_empty:
        return out
    x0, y0, x1, y1 = geom.bounds
    c0, c1 = max(int(x0), 0), min(int(x1) + 1, shape[1])
    r0, r1 = max(int(y0), 0), min(int(y1) + 1, shape[0])
    if c1 <= c0 or r1 <= r0:
        return out
    cc, rr = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
    out[r0:r1, c0:c1] = shapely.contains_xy(geom, cc, rr)
    return out


def dndvi_of(geom, all_geoms, ndvi: np.ndarray) -> float:
    """폴리곤 내부 NDVI 평균 - 주변링(3~8px = 30~80m, 타 개소 제외) NDVI 평균."""
    inside = rasterize(geom, ndvi.shape)
    ring_g = geom.buffer(8).difference(geom.buffer(3))
    ring_g = ring_g.difference(unary_union([g.buffer(3) for g in all_geoms]))
    ring = rasterize(ring_g, ndvi.shape)
    if not inside.any() or not ring.any():
        return 0.0
    return float(ndvi[inside].mean() - ndvi[ring].mean())



# ---------- learned temporal presence gate ----------

def _stats(a, m):
    z = a[m]; z = z[np.isfinite(z)]
    if not len(z): return np.zeros(7, np.float32)
    return np.array([np.mean(z), np.std(z), *np.quantile(z, [.1,.25,.5,.75,.9])], np.float32)

def _raw_features(bands, inside, ring, month):
    x=bands.astype(np.float32)/10000; valid=(x[...,0]>0)&(x[...,3]>0)
    red,green,blue,nir=(x[...,i] for i in range(4))
    ndvi=(nir-red)/np.maximum(nir+red,1e-5); ndwi=(green-nir)/np.maximum(green+nir,1e-5)
    bsi=((red+blue)-(nir+green))/np.maximum(red+blue+nir+green,1e-5)
    f=[]
    for v in (red,green,blue,nir,ndvi,ndwi,bsi):
        si,sr=_stats(v,inside&valid),_stats(v,ring&valid); f.extend(si);f.extend(sr);f.extend(si-sr)
    f += [inside.sum()/100,(inside&valid).sum()/max(inside.sum(),1),np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12)]
    return np.asarray(f,np.float32)

def _temporalize(raw):
    raw=np.asarray(raw,np.float32); med=np.median(raw,axis=0); q25,q75=np.quantile(raw,[.25,.75],axis=0)
    ranks=np.argsort(np.argsort(raw,axis=0),axis=0)/max(len(raw)-1,1)
    return np.concatenate([raw,(raw-med)/np.maximum(q75-q25,1e-4),ranks],axis=1).astype(np.float32)

def predict_presence(entry, bundle):
    dates=sorted(entry["dates"]); sites=entry["sites"]
    geoms=[polygons_to_geom(s.get("prior_polygon")) for s in sites]
    ans={d:{} for d in dates}
    for s,g in zip(sites,geoms):
        if g is None:
            for d in dates: ans[d][s["site_id"]]=False
            continue
        inside=rasterize(g,(256,256))
        others=unary_union([q.buffer(3) for q in geoms if q is not None and q is not g])
        ring=rasterize(g.buffer(8).difference(g.buffer(2)).difference(others),(256,256))
        raw=[_raw_features(np.load(entry["dates"][d]),inside,ring,int(d[4:6])) for d in dates]
        prob=bundle["model"].predict_proba(_temporalize(raw))[:,1]
        # Suppress isolated cloudy/no-data dates with a three-date local smoother.
        alpha=bundle.get("smooth_alpha",.45)
        local=np.asarray([prob[max(0,j-1):min(len(prob),j+2)].mean() for j in range(len(prob))])
        prob=(1-alpha)*prob+alpha*local
        for d,pr in zip(dates,prob): ans[d][s["site_id"]]=bool(pr>=bundle["threshold"])
    return ans

# ---------- 모델 ----------

def load_model(device: str):
    from transformers import Sam3TrackerModel

    source = str(SAM3_DIR) if (SAM3_DIR / "config.json").exists() else "facebook/sam3"
    print(f"[model] {source} + {CKPT.name} on {device}")
    model = Sam3TrackerModel.from_pretrained(source, dtype=torch.float32)
    sd = torch.load(CKPT, weights_only=True, map_location="cpu")
    model.prompt_encoder.load_state_dict(sd["prompt_encoder"])
    model.mask_decoder.load_state_dict(sd["mask_decoder"])
    return model.to(device).eval()


def predict_patch(model, device, bands: np.ndarray, sites: list, presence: dict) -> dict:
    """패치 1장(창 x 시점) -> {site_id: 3중 배열 폴리곤(빈 배열 = 안 보임)}."""
    hints = [(s["site_id"], polygons_to_geom(s.get("prior_polygon"))) for s in sites]
    all_geoms = [g for _, g in hints if g is not None]

    img = render_rgb(bands)
    nir = bands[:, :, 3].astype(np.float32)   # B08
    red = bands[:, :, 0].astype(np.float32)   # B04
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)

    # 1) 인코딩 1회
    x = torch.from_numpy(
        np.array(img.resize((MODEL_IN, MODEL_IN), Image.BILINEAR), dtype=np.float32) / 255.0
    )
    x = ((x - 0.5) / 0.5).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        enc = model(
            pixel_values=x,
            input_points=torch.zeros(1, 1, 1, 2, device=device),
            input_labels=-torch.ones(1, 1, 1, dtype=torch.int32, device=device),
            multimask_output=False,
        )
        enc_flip = model(
            pixel_values=torch.flip(x,(-1,)),
            input_points=torch.zeros(1,1,1,2,device=device),
            input_labels=-torch.ones(1,1,1,dtype=torch.int32,device=device),
            multimask_output=False,
        )

    # 2) 개소별 박스 + 대표점 프롬프트 -> 디코더 -> 폴리곤 (좌표는 256px 프레임)
    k = MODEL_IN / IMG
    out_polys = {}
    for sid, g in hints:
        if g is None:
            out_polys[sid] = []   # 힌트 없음 — 판정 불가, 안 보임 처리
            continue
        x0, y0, x1, y1 = (v * UP for v in g.bounds)
        if not presence.get(sid, False):
            out_polys[sid] = []
            continue
        w, h = x1 - x0, y1 - y0
        x0, y0, x1, y1 = x0 - PROMPT_PAD*w, y0 - PROMPT_PAD*h, x1 + PROMPT_PAD*w, y1 + PROMPT_PAD*h
        rep = g.representative_point()
        px, py = rep.x * UP, rep.y * UP

        # Point+box was slightly more accurate than cross-prompt score selection.
        box_prompt=torch.tensor([[[x0*k,y0*k,x1*k,y1*k]]],device=device)
        with torch.no_grad():
            original=model(image_embeddings=enc.image_embeddings,multimask_output=True,
                input_boxes=box_prompt,input_points=torch.tensor([[[[px*k,py*k]]]],device=device),
                input_labels=torch.tensor([[[1]]],dtype=torch.int32,device=device))
            oi=int(original.iou_scores[0,0].argmax()); original_logits=original.pred_masks[0,0,oi]
            flipped=model(image_embeddings=enc_flip.image_embeddings,multimask_output=True,
                input_boxes=torch.tensor([[[MODEL_IN-x1*k,y0*k,MODEL_IN-x0*k,y1*k]]],device=device),
                input_points=torch.tensor([[[[MODEL_IN-px*k,py*k]]]],device=device),
                input_labels=torch.tensor([[[1]]],dtype=torch.int32,device=device))
            fi=int(flipped.iou_scores[0,0].argmax()); flipped_logits=flipped.pred_masks[0,0,fi]
        logits=F.interpolate(original_logits[None,None].float(),size=(IMG,IMG),mode="bilinear")[0,0]
        logits_flip=F.interpolate(flipped_logits[None,None].float(),size=(IMG,IMG),mode="bilinear")[0,0].flip(-1)
        mask=((logits+logits_flip)/2).cpu().numpy()>1.0

        # 최소면적: m2 기준 max(100, min(200, 0.3 x 힌트면적)) 을 px2 로 환산
        min_area_px2 = max(100.0, min(200.0, 0.3 * g.area * M2_PER_PX2)) / M2_PER_PX2
        polys = px_polys_from_mask(mask, min_area_px2)

        # OOF-tuned temporal multispectral classifier is the visibility gate.
        # High-fidelity contour preservation: 0.5m tolerance instead of 2.5m blunt simplify
        merged = unary_union(polys).simplify(0.5 / (M2_PER_PX2 ** 0.5)) if polys else None
        out_polys[sid] = geom_to_polys(merged)
    return out_polys
