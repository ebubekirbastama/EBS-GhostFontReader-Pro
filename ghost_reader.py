#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EBS Ghost Font Reader v1 - zamansal hareket tabanlı metin görünürleştirme.

Tek kare OCR yapmaz. Video boyunca hareket, varyans, zamansal gradyan ve
optik akış bilgilerini birleştirir; ardından çoklu netleştirme adayları üretir.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

ProgressFn = Optional[Callable[[int, str], None]]


@dataclass
class Config:
    input: str
    output: str = "ghost_output"
    roi: Optional[Tuple[int, int, int, int]] = None
    max_width: int = 1200
    frame_step: int = 1
    max_frames: int = 900
    stabilize: bool = True
    vertical_compensate: bool = True
    vertical_max_step: int = 14
    vertical_smooth: int = 5
    upscale: int = 3
    ocr: bool = False
    lang: str = "eng"


def progress(cb: ProgressFn, value: int, text: str) -> None:
    if cb:
        cb(max(0, min(100, value)), text)


def normalize_u8(a: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    a = np.nan_to_num(a.astype(np.float32))
    p1, p2 = np.percentile(a, [lo, hi])
    if p2 <= p1 + 1e-9:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - p1) * (255.0 / (p2 - p1)), 0, 255).astype(np.uint8)


def resize_keep(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    s = max_width / float(w)
    return cv2.resize(frame, (max_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


def default_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    return int(w * .20), int(h * .20), int(w * .60), int(h * .60)


def safe_crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = map(int, roi)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Geçersiz seçim alanı: {roi}")
    return frame[y1:y2, x1:x2]


def read_video(cfg: Config, cb: ProgressFn = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(cfg.input)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {cfg.input}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or cfg.max_frames
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frames: list[np.ndarray] = []
    raw = 0
    while len(frames) < cfg.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if raw % max(1, cfg.frame_step) == 0:
            frame, _ = resize_keep(frame, cfg.max_width)
            if cfg.roi:
                frame = safe_crop(frame, cfg.roi)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        raw += 1
        if raw % 10 == 0:
            progress(cb, int(min(25, 25 * raw / max(total, 1))), f"Video okunuyor: {len(frames)} kare")
    cap.release()
    if len(frames) < 3:
        raise RuntimeError("Analiz için en az 3 kare gerekli.")
    return frames, fps


def ecc_stabilize(frames: list[np.ndarray], cb: ProgressFn = None) -> list[np.ndarray]:
    ref = cv2.GaussianBlur(frames[0], (5, 5), 0)
    out = [frames[0]]
    for i, cur0 in enumerate(frames[1:], 1):
        cur = cv2.GaussianBlur(cur0, (5, 5), 0)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
            cv2.findTransformECC(ref, cur, warp, cv2.MOTION_TRANSLATION, criteria, None, 3)
            aligned = cv2.warpAffine(cur0, warp, (cur0.shape[1], cur0.shape[0]),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT)
        except cv2.error:
            aligned = cur0
        out.append(aligned)
        if i % 10 == 0:
            progress(cb, 25 + int(10 * i / len(frames)), f"Kareler sabitleniyor: {i}/{len(frames)}")
    return out




def _vertical_signature(frame: np.ndarray) -> np.ndarray:
    """Harf bandının dikey konumunu izlemek için 1B satır imzası üretir.

    Sabit sütun gürültüsünü azaltır, yatay yöndeki harf kenarlarını öne çıkarır.
    """
    f = frame.astype(np.float32)
    # Sütunlara özgü sabit dokuyu ve yavaş aydınlatma değişimini bastır.
    f -= np.median(f, axis=0, keepdims=True)
    f = cv2.GaussianBlur(f, (0, 0), 0.8)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    # Harflerin bulunduğu satırlarda yatay kenar enerjisi artar.
    sig = np.mean(np.abs(gx), axis=1)
    sig = cv2.GaussianBlur(sig.reshape(-1, 1), (1, 0), 2.0).ravel()
    sig -= np.median(sig)
    scale = np.std(sig) + 1e-6
    return sig / scale


def _best_1d_shift(reference: np.ndarray, current: np.ndarray, max_shift: int) -> tuple[int, float]:
    """current imzasını reference ile hizalayan dikey kaymayı döndürür."""
    best_shift, best_score = 0, -1e30
    n = min(len(reference), len(current))
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = reference[shift:n]
            b = current[:n-shift]
        else:
            a = reference[:n+shift]
            b = current[-shift:n]
        if len(a) < 12:
            continue
        den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        score = float(np.dot(a, b) / den)
        if score > best_score:
            best_shift, best_score = shift, score
    return best_shift, best_score


def compensate_vertical_drift(
    frames: list[np.ndarray], max_step: int = 14, smooth: int = 5, cb: ProgressFn = None
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Harf katmanının kareler boyunca yukarı-aşağı sürüklenmesini telafi eder.

    Karelerin satır imzaları ardışık olarak eşleştirilir. Elde edilen kümülatif
    hareket yumuşatılır ve her kare ters yönde kaydırılır.
    """
    if len(frames) < 3:
        return frames, [0.0] * len(frames), [1.0] * len(frames)

    signatures = [_vertical_signature(f) for f in frames]
    cumulative = [0.0]
    scores = [1.0]
    for i in range(1, len(frames)):
        shift, score = _best_1d_shift(signatures[i-1], signatures[i], max_step)
        # Düşük güvenli eşleşmelerde ani sıçramayı engelle.
        if score < 0.05:
            shift = 0
        cumulative.append(cumulative[-1] + float(shift))
        scores.append(score)
        if i % 10 == 0:
            progress(cb, 32 + int(8 * i / len(frames)), f"Dikey harf hareketi izleniyor: {i}/{len(frames)-1}")

    trajectory = np.asarray(cumulative, np.float32)
    # Uç değerleri bastır ve hareket yolunu yumuşat.
    if smooth > 1:
        k = max(3, int(smooth) | 1)
        trajectory = cv2.GaussianBlur(trajectory.reshape(-1, 1), (1, k), 0).ravel()
    trajectory -= np.median(trajectory)

    aligned: list[np.ndarray] = []
    h, w = frames[0].shape
    for frame, y in zip(frames, trajectory):
        # İzlenen hareket +y ise içeriği -y kaydırarak harfleri sabitle.
        m = np.float32([[1, 0, 0], [0, 1, -float(y)]])
        aligned.append(cv2.warpAffine(
            frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        ))
    return aligned, trajectory.tolist(), scores


def save_vertical_diagnostics(out: Path, trajectory: list[float], scores: list[float]) -> None:
    rows = ["frame,vertical_shift,match_score"]
    rows += [f"{i},{y:.4f},{s:.6f}" for i, (y, s) in enumerate(zip(trajectory, scores))]
    (out / "vertical_motion.csv").write_text("\n".join(rows), encoding="utf-8")

    # Harici grafik bağımlılığı olmadan basit hareket izi görseli.
    width, height = 900, 260
    canvas = np.full((height, width, 3), 255, np.uint8)
    if trajectory:
        arr = np.asarray(trajectory, np.float32)
        lim = max(1.0, float(np.max(np.abs(arr))))
        xs = np.linspace(20, width - 20, len(arr)).astype(np.int32)
        ys = (height / 2 - arr / lim * (height * 0.38)).astype(np.int32)
        cv2.line(canvas, (20, height // 2), (width - 20, height // 2), (180, 180, 180), 1)
        for i in range(1, len(xs)):
            cv2.line(canvas, (int(xs[i-1]), int(ys[i-1])), (int(xs[i]), int(ys[i])), (0, 0, 0), 2)
        cv2.putText(canvas, f"Vertical drift: +/-{lim:.1f} px", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "00_vertical_motion_trace.png"), canvas)

def remove_banding(img: np.ndarray) -> np.ndarray:
    """Satır/sütun bantlarını bastırır; harf yapısını korur."""
    f = img.astype(np.float32)
    row = cv2.GaussianBlur(np.median(f, axis=1).reshape(-1, 1), (1, 0), 5)
    col = cv2.GaussianBlur(np.median(f, axis=0).reshape(1, -1), (0, 1), 5)
    corrected = f - 0.45 * row - 0.25 * col + 0.70 * np.median(f)
    return corrected


def unsharp(img: np.ndarray, amount: float = 1.8, sigma: float = 2.0) -> np.ndarray:
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    return np.clip(f + amount * (f - blur), 0, 255).astype(np.uint8)


def clean_binary(img: np.ndarray, invert: bool = False) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, 31, 3)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k2)
    return bw


def crop_content(img: np.ndarray, margin: int = 8) -> np.ndarray:
    # Harici koyu/açık çerçeveleri kırpmak için gradyan tabanlı içerik sınırı.
    g = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    score = np.mean(np.abs(g), axis=1)
    threshold = np.percentile(score, 35)
    ys = np.where(score > threshold)[0]
    if len(ys) < 10:
        return img
    y1, y2 = max(0, int(ys[0]) - margin), min(img.shape[0], int(ys[-1]) + margin + 1)
    return img[y1:y2]


def analyze(cfg: Config, cb: ProgressFn = None) -> dict:
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    progress(cb, 1, "Analiz başlatıldı")
    frames, fps = read_video(cfg, cb)
    original_first = frames[0]
    if cfg.stabilize:
        frames = ecc_stabilize(frames, cb)
    vertical_trajectory = [0.0] * len(frames)
    vertical_scores = [1.0] * len(frames)
    if cfg.vertical_compensate:
        frames, vertical_trajectory, vertical_scores = compensate_vertical_drift(
            frames, cfg.vertical_max_step, cfg.vertical_smooth, cb
        )
        save_vertical_diagnostics(out, vertical_trajectory, vertical_scores)
        cv2.imwrite(str(out / "00_roi_aligned_first_frame.png"), frames[0])
    progress(cb, 41, "Zamansal özellikler hesaplanıyor")

    stack = np.stack(frames).astype(np.float32)
    # Temel zamansal özellikler
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    p10 = np.percentile(stack, 10, axis=0)
    p90 = np.percentile(stack, 90, axis=0)
    robust_range = p90 - p10
    diffs = np.abs(np.diff(stack, axis=0))
    diff_mean = diffs.mean(axis=0)
    diff_rms = np.sqrt(np.mean(np.diff(stack, axis=0) ** 2, axis=0))

    # İkinci türev: periyodik titreşimi ve hızlanan/yavaşlayan noktaları vurgular.
    if len(frames) >= 4:
        second = np.diff(stack, n=2, axis=0)
        accel_rms = np.sqrt(np.mean(second ** 2, axis=0))
    else:
        accel_rms = diff_rms

    # Optik akış enerji ve yön tutarlılığı
    h, w = frames[0].shape
    flow_energy = np.zeros((h, w), np.float32)
    sx = np.zeros((h, w), np.float32)
    sy = np.zeros((h, w), np.float32)
    for i in range(1, len(frames)):
        a = cv2.GaussianBlur(frames[i-1], (5, 5), 0)
        b = cv2.GaussianBlur(frames[i], (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 19, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        mag = cv2.magnitude(dx, dy)
        # Aşırı akış değerlerini kırp; ekran kaydı sıkıştırma artefaktlarını azaltır.
        cap = np.percentile(mag, 98.5)
        mag = np.minimum(mag, cap)
        flow_energy += mag
        sx += dx
        sy += dy
        if i % 10 == 0:
            progress(cb, 44 + int(21 * i / len(frames)), f"Optik akış: {i}/{len(frames)-1}")
    coherence = np.sqrt(sx*sx + sy*sy) / (flow_energy + 1e-6)
    coherent = flow_energy * np.clip(coherence, 0, 1)

    raw_maps = {
        "01_temporal_std.png": std,
        "02_robust_range.png": robust_range,
        "03_difference_mean.png": diff_mean,
        "04_difference_rms.png": diff_rms,
        "05_acceleration_rms.png": accel_rms,
        "06_flow_energy.png": flow_energy,
        "07_flow_coherence.png": coherence,
        "08_coherent_flow.png": coherent,
    }
    cv2.imwrite(str(out / "00_roi_first_frame.png"), original_first)
    for name, arr in raw_maps.items():
        cv2.imwrite(str(out / name), normalize_u8(arr))

    progress(cb, 68, "Gürültü ve bantlar temizleniyor")
    # Birbirinden farklı üç füzyon; video tipine göre biri diğerinden iyi olabilir.
    n_std = normalize_u8(remove_banding(std))
    n_diff = normalize_u8(remove_banding(diff_rms))
    n_acc = normalize_u8(remove_banding(accel_rms))
    n_flow = normalize_u8(remove_banding(flow_energy))
    n_coh = normalize_u8(remove_banding(coherent))

    fusion_a = cv2.addWeighted(n_diff, .48, n_flow, .32, 0)
    fusion_a = cv2.addWeighted(fusion_a, .80, n_std, .20, 0)
    fusion_b = cv2.addWeighted(n_acc, .50, n_coh, .30, 0)
    fusion_b = cv2.addWeighted(fusion_b, .80, n_diff, .20, 0)
    fusion_c = np.maximum(n_diff, n_flow).astype(np.uint8)

    candidates = {
        "09_fusion_balanced.png": fusion_a,
        "10_fusion_periodic.png": fusion_b,
        "11_fusion_max.png": fusion_c,
    }
    ocr_inputs: list[tuple[str, np.ndarray]] = []
    scale = max(1, int(cfg.upscale))
    for idx, (name, base) in enumerate(candidates.items(), 1):
        base = crop_content(base)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(base)
        sharp = unsharp(clahe, 2.1, 1.6)
        if scale > 1:
            sharp = cv2.resize(sharp, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            sharp = unsharp(sharp, 1.4, 1.2)
        stem = Path(name).stem
        cv2.imwrite(str(out / f"{stem}_sharp.png"), sharp)
        bw = clean_binary(sharp, invert=False)
        bwi = clean_binary(sharp, invert=True)
        cv2.imwrite(str(out / f"{stem}_binary.png"), bw)
        cv2.imwrite(str(out / f"{stem}_binary_inverted.png"), bwi)
        ocr_inputs.extend([(stem+"_gray", sharp), (stem+"_bw", bw), (stem+"_bwi", bwi)])

    # Kullanıcı açısından en kolay okunur ana çıktı.
    best = crop_content(fusion_a)
    best = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(best)
    best = unsharp(best, 2.3, 1.8)
    best = cv2.resize(best, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    best = unsharp(best, 1.5, 1.1)
    cv2.imwrite(str(out / "12_BEST_READABLE.png"), best)
    cv2.imwrite(str(out / "13_BEST_BINARY.png"), clean_binary(best))
    cv2.imwrite(str(out / "14_BEST_BINARY_INVERTED.png"), clean_binary(best, True))

    # Karşılaştırma sayfası
    thumbs = []
    for title, img in [("STD", n_std), ("DIFF", n_diff), ("FLOW", n_flow),
                       ("BALANCED", fusion_a), ("PERIODIC", fusion_b), ("MAX", fusion_c)]:
        im = cv2.resize(img, (480, 220), interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2, cv2.LINE_AA)
        thumbs.append(im)
    montage = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:])])
    cv2.imwrite(str(out / "15_comparison_montage.png"), montage)

    result = {
        "input": cfg.input,
        "output": str(out.resolve()),
        "roi": cfg.roi,
        "frames_analyzed": len(frames),
        "fps": fps,
        "best_image": str((out / "12_BEST_READABLE.png").resolve()),
        "ocr_text": "",
        "ocr_candidates": [],
        "vertical_compensation": cfg.vertical_compensate,
        "vertical_shift_min": float(min(vertical_trajectory)) if vertical_trajectory else 0.0,
        "vertical_shift_max": float(max(vertical_trajectory)) if vertical_trajectory else 0.0,
        "vertical_match_score_mean": float(np.mean(vertical_scores)) if vertical_scores else 0.0,
    }

    if cfg.ocr:
        progress(cb, 90, "OCR adayları deneniyor")
        try:
            import pytesseract
            variants = ocr_inputs + [("best", best), ("best_bw", clean_binary(best)),
                                     ("best_bwi", clean_binary(best, True))]
            seen = set()
            for name, img in variants:
                for psm in (7, 8, 6, 13):
                    text = pytesseract.image_to_string(
                        img, lang=cfg.lang,
                        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
                    ).strip()
                    key = " ".join(text.split())
                    if key and key not in seen:
                        seen.add(key)
                        result["ocr_candidates"].append({"source": name, "psm": psm, "text": key})
            if result["ocr_candidates"]:
                # Harf/rakam yoğunluğu yüksek, kısa tek satırlı adayları tercih et.
                def score(item):
                    t = item["text"]
                    alnum = sum(ch.isalnum() for ch in t)
                    noise = sum(not (ch.isalnum() or ch.isspace()) for ch in t)
                    return alnum * 3 - noise * 2 - abs(len(t) - alnum)
                result["ocr_candidates"].sort(key=score, reverse=True)
                result["ocr_text"] = result["ocr_candidates"][0]["text"]
            (out / "ocr_candidates.txt").write_text(
                "\n".join(f"[{x['source']} psm={x['psm']}] {x['text']}" for x in result["ocr_candidates"]),
                encoding="utf-8"
            )
        except Exception as exc:
            result["ocr_error"] = str(exc)

    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(cb, 100, "Tamamlandı")
    return result


def parse_roi(value: str) -> tuple[int, int, int, int]:
    vals = tuple(int(v.strip()) for v in value.split(","))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("ROI x,y,w,h biçiminde olmalı")
    return vals


def main() -> int:
    p = argparse.ArgumentParser(description="Ghost Font Reader v4")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="ghost_output")
    p.add_argument("--roi", type=parse_roi)
    p.add_argument("--max-width", type=int, default=1200)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=900)
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--no-vertical-compensation", action="store_true")
    p.add_argument("--vertical-max-step", type=int, default=14)
    p.add_argument("--vertical-smooth", type=int, default=5)
    p.add_argument("--no-stabilize", action="store_true")
    p.add_argument("--ocr", action="store_true")
    p.add_argument("--lang", default="eng")
    a = p.parse_args()
    cfg = Config(input=a.input, output=a.output, roi=a.roi, max_width=a.max_width,
                 frame_step=a.frame_step, max_frames=a.max_frames,
                 stabilize=not a.no_stabilize,
                 vertical_compensate=not a.no_vertical_compensation,
                 vertical_max_step=a.vertical_max_step, vertical_smooth=a.vertical_smooth,
                 upscale=a.upscale, ocr=a.ocr, lang=a.lang)
    result = analyze(cfg, lambda n, s: print(f"[{n:3d}%] {s}"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
