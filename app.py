"""
Symbol Detection Pro — Enhanced v4.3 (Pro UI + Softer Defaults + Recover Pass, NO auto-save)
Upload PDF/PNG → Crop Symbols → SIFT + Template Matching → Verify → View → Download PDF

What's changed vs v4.2 to fix "not detecting any symbols":
  • Softer defaults: Verify threshold 0.66 (was 0.70), Cross-template margin 0.03 (was 0.06)
  • Adaptive TM threshold now based on (maxVal - delta) with a small delta so peaks survive
  • Lower SIFT minimum matches (8)
  • Template auto-trim of white borders (helps when crops include background)
  • Recover pass: if first pass finds 0, re-verify with lower threshold & no margin (still filtered)

Still:
  • Per-template colored boxes + legend
  • Global NMS across templates
  • Result kept in memory; only saved when you click Download PDF
Requirements:
  pip install opencv-contrib-python dash dash-bootstrap-components plotly pymupdf pillow
"""

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from pathlib import Path
from datetime import datetime
import base64
import fitz  # PyMuPDF
from PIL import Image  # for PDF export

# -----------------------------------
# Configuration / dirs
# -----------------------------------
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("cropped_symbols")
RESULTS_DIR = Path("output")
for d in (UPLOAD_DIR, OUTPUT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Global state
# -----------------------------------
app_state = {
    'current_image': None,          # RGB np.array
    'current_image_path': None,     # original file path (pdf/png)
    'img_width': 0,
    'img_height': 0,
    'coords': [],                   # last drawn rect (dict)
    'saved_crops': [],              # list of {id, filename, filepath, coords, size}
    'crop_counter': 1,
    'detection_results': None       # {'result_image': np.array, ...}
}

# -----------------------------------
# App (Pro UI theme)
# -----------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Symbol Detection Pro"
server = app.server

# -----------------------------------
# UI Helpers
# -----------------------------------
def app_header():
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(
                        src="https://cdn-icons-png.flaticon.com/512/992/992651.png",
                        height="28px"
                    ), width="auto"),
                    dbc.Col(dbc.NavbarBrand("Symbol Detection Pro", className="ms-2")),
                ], align="center", className="g-1"),
                href="#",
                style={"textDecoration": "none"}
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        ]),
        color="primary",
        dark=True,
        className="mb-4 shadow-sm",
    )

def step_card(title, body_children):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(title, className="fw-semibold"),
            ], className="d-flex align-items-center fs-5 mb-2"),
            *([body_children] if isinstance(body_children, (html.Div, dcc.Graph)) else body_children)
        ]),
        className="shadow-sm"
    )

# -----------------------------------
# Utils
# -----------------------------------
def pdf_to_image(pdf_path, dpi=300, page_num=0):
    """Convert a single PDF page to an RGB image (np.array)."""
    doc = fitz.open(pdf_path)
    try:
        if page_num >= len(doc):
            page_num = 0
        page = doc[page_num]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    finally:
        doc.close()

def create_empty_figure():
    fig = go.Figure()
    fig.add_annotation(
        text="Upload a PDF or PNG file to begin",
        showarrow=False,
        font=dict(size=20, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=560,
        plot_bgcolor='#f8f9fa',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def create_figure_with_image(img_rgb, rectangles=None):
    fig = px.imshow(img_rgb)
    if rectangles:
        for rect in rectangles:
            x0, y0, x1, y1 = rect['coords']
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="lime", width=2),
                fillcolor="rgba(0,255,0,0.10)"
            )
            fig.add_annotation(
                x=x0, y=max(0, y0-5),
                text=f"#{rect['id']}",
                showarrow=False,
                font=dict(color="lime", size=12),
                bgcolor="rgba(0,0,0,0.5)"
            )
    fig.update_layout(
        dragmode="drawrect",
        newshape=dict(line_color="cyan", line_width=2),
        height=760,
        title="Step 2 • Crop Symbol Templates (draw rectangles)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def parse_last_rect_from_relayout(relayout_data):
    """Return (x1, y1, x2, y2) sorted, or None."""
    if not relayout_data:
        return None
    if "shapes" in relayout_data and relayout_data["shapes"]:
        s = relayout_data["shapes"][-1]
        x0, y0 = float(s["x0"]), float(s["y0"])
        x1, y1 = float(s["x1"]), float(s["y1"])
    else:
        keys = [k for k in relayout_data if k.startswith("shapes[") and (k.endswith("].x0") or k.endswith("].x1"))]
        if not keys:
            return None
        idx = max(int(k.split('[')[1].split(']')[0]) for k in keys)
        try:
            x0 = float(relayout_data.get(f"shapes[{idx}].x0"))
            y0 = float(relayout_data.get(f"shapes[{idx}].y0"))
            x1 = float(relayout_data.get(f"shapes[{idx}].x1"))
            y1 = float(relayout_data.get(f"shapes[{idx}].y1"))
        except Exception:
            return None
    xa, xb = sorted([int(round(x0)), int(round(x1))])
    ya, yb = sorted([int(round(y0)), int(round(y1))])
    return xa, ya, xb, yb

def edge_map(gray):
    g = cv2.GaussianBlur(gray, (3,3), 0)
    e = cv2.Canny(g, 50, 150)
    return e.astype(np.float32) / 255.0

def topk_peaks_from_response(resp, k=60, thr=0.6, min_dist=10):
    """Pick top-k local peaks from a template-match response."""
    if resp is None:
        return []
    resp = resp.copy()
    H, W = resp.shape
    peaks = []
    for _ in range(k):
        _, maxVal, _, maxLoc = cv2.minMaxLoc(resp)
        if maxVal < thr:
            break
        y, x = maxLoc[1], maxLoc[0]
        peaks.append((y, x, float(maxVal)))
        y0, y1 = max(0, y - min_dist), min(H, y + min_dist + 1)
        x0, x1 = max(0, x - min_dist), min(W, x + min_dist + 1)
        resp[y0:y1, x0:x1] = -1.0
    return peaks

def draw_template_strip(templates_bgr, strip_height=120, pad=12, colors_rgb=None):
    """Horizontal strip of template thumbnails (BGR) with optional colored border."""
    thumbs = []
    color_idx = 0
    for tbgr in templates_bgr:
        if tbgr is None or tbgr.size == 0:
            continue
        h, w = tbgr.shape[:2]
        s = strip_height / max(1, h)
        tw, th = int(w*s), strip_height
        thumb = cv2.resize(tbgr, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if colors_rgb is not None and len(colors_rgb) > 0:
            rr, gg, bb = colors_rgb[color_idx % len(colors_rgb)]
            cv2.rectangle(thumb, (0,0), (tw-1, th-1), (bb, gg, rr), 3)  # RGB→BGR
            color_idx += 1
        thumbs.append(thumb)
        thumbs.append(np.full((strip_height, pad, 3), 255, np.uint8))
    if not thumbs:
        return None
    return np.concatenate(thumbs, axis=1)

def non_max_suppression(detections, iou_threshold=0.35):
    if len(detections) == 0:
        return []
    boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
    scores = np.array([d['confidence'] for d in detections], dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]

def downscale_for_detection(img, max_side=3000, max_pixels=3_000_000):
    """Uniformly downscale image to be memory-safe. Returns (resized, scale_factor)."""
    h, w = img.shape[:2]
    r_side = max_side / max(h, w)
    r_pix  = (max_pixels / (h * w)) ** 0.5
    r = min(1.0, r_side, r_pix)
    if r < 1.0:
        new = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
    else:
        new = img
    return new, r

def dice_coeff(maskA, maskB):
    inter = np.logical_and(maskA, maskB).sum()
    size = maskA.sum() + maskB.sum()
    return (2.0 * inter / size) if size > 0 else 0.0

def hu_similarity(maskA, maskB):
    mA = cv2.moments(maskA.astype(np.uint8))
    mB = cv2.moments(maskB.astype(np.uint8))
    hA = cv2.HuMoments(mA).flatten()
    hB = cv2.HuMoments(mB).flatten()
    hA = np.sign(hA) * np.log10(np.abs(hA) + 1e-30)
    hB = np.sign(hB) * np.log10(np.abs(hB) + 1e-30)
    dist = np.linalg.norm(hA - hB)
    return float(1.0 / (1.0 + dist))  # larger is better

def get_palette_rgb(n):
    base = [
        (31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),
        (140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207)
    ]
    if n <= len(base):
        return base[:n]
    out = base[:]
    i = 0
    while len(out) < n:
        r,g,b = base[i % len(base)]
        out.append(((r+40) % 256, (g+20) % 256, (b+60) % 256))
        i += 1
    return out[:n]

def trim_white_border(gray, tol=245):
    """Trim near-white border from a template crop. Returns cropped gray."""
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    mask = gray < tol
    if not np.any(mask):
        return gray
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    pad = 1
    y1 = max(0, y1 - pad); x1 = max(0, x1 - pad)
    y2 = min(gray.shape[0]-1, y2 + pad); x2 = min(gray.shape[1]-1, x2 + pad)
    return gray[y1:y2+1, x1:x2+1]

# -----------------------------------
# Detector (v4) + Softer Defaults + Recover Pass (no auto-save)
# -----------------------------------
def run_symbol_detection(image_path, template_paths, verify_min=0.66, margin_other=0.03):
    """
    Returns in-memory result image; draws boxes with per-template colors.
    verify_min: final verification threshold (0..1)
    margin_other: own intensity NCC must beat best other template by at least this margin
    """
    # --- Load original
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        if app_state['current_image'] is None:
            return None
        img_bgr = cv2.cvtColor(app_state['current_image'], cv2.COLOR_RGB2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Normalize for matching
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray_eq = clahe.apply(img_gray)

    # Downscale working copies
    img_gray_eq_s, r_img = downscale_for_detection(img_gray_eq, max_side=3000, max_pixels=3_000_000)
    img_edge_s = cv2.Canny(cv2.GaussianBlur(img_gray_eq_s, (3,3), 0), 50, 150).astype(np.float32) / 255.0
    inv_r = (1.0 / r_img) if r_img > 0 else 1.0

    # Load & prep templates (scaled to working scale)
    templates = []
    templates_bgr = []
    for p in template_paths:
        tbgr = cv2.imread(str(p))
        if tbgr is None:
            continue
        tgray0 = cv2.cvtColor(tbgr, cv2.COLOR_BGR2GRAY)
        tgray0 = trim_white_border(tgray0, tol=245)  # helps weak responses
        tgray = clahe.apply(tgray0)
        if r_img < 1.0:
            tgray = cv2.resize(tgray, (int(tgray.shape[1]*r_img), int(tgray.shape[0]*r_img)),
                               interpolation=cv2.INTER_AREA)
        templates.append({'path': p, 'bgr': tbgr, 'gray_s': tgray})
        templates_bgr.append(tbgr)
    if not templates:
        return None

    n_templates = len(templates)
    # SIFT/FLANN
    try:
        sift = cv2.SIFT_create(nfeatures=4000)
    except Exception:
        sift = None
    flann = None
    kp_img, des_img = None, None
    if sift is not None:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        kp_img, des_img = sift.detectAndCompute(img_gray_eq_s, None)

    # Search params
    SCALES    = [0.75, 0.9, 1.0, 1.15, 1.3]
    ROTATIONS = [-20, -10, 0, 10, 20]
    TM_BASE   = 0.58  # softer base
    TM_THR    = min(0.78, TM_BASE + 0.01*max(0, n_templates-1))  # lighter bump
    EDGE_W    = 0.5
    INT_W     = 0.5
    SIFT_MIN_MATCH = 8  # lower to allow more SIFT proposals
    FUSE_W_SIFT    = 0.30
    IOU_NMS        = 0.35

    Hs, Ws = img_gray_eq_s.shape[:2]
    all_dets_orig = []  # boxes mapped to original coords, with raw TM score stored

    for t_idx, T in enumerate(templates):
        t_s = T['gray_s']
        th0, tw0 = t_s.shape[:2]

        # SIFT (global, scaled plane)
        if sift is not None and des_img is not None and th0 >= 12 and tw0 >= 12:
            kp_t, des_t = sift.detectAndCompute(t_s, None)
            if des_t is not None and len(kp_t) >= 6:
                matches = flann.knnMatch(des_t, des_img, k=2)
                good = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.72 * n.distance:
                            good.append(m)
                if len(good) >= SIFT_MIN_MATCH:
                    src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                    dst = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                    if Hm is not None:
                        box = np.float32([[0,0],[tw0,0],[tw0,th0],[0,th0]]).reshape(-1,1,2)
                        proj = cv2.perspectiveTransform(box, Hm).reshape(-1,2)
                        x1s = max(0, int(np.min(proj[:,0])))
                        y1s = max(0, int(np.min(proj[:,1])))
                        x2s = min(Ws-1, int(np.max(proj[:,0])))
                        y2s = min(Hs-1, int(np.max(proj[:,1])))
                        if x2s > x1s and y2s > y1s:
                            inliers = int(mask.sum()) if mask is not None else len(good)
                            sift_conf = float(np.clip(inliers / max(1, len(good)), 0, 1))
                            x1 = int(round(x1s * inv_r)); y1 = int(round(y1s * inv_r))
                            x2 = int(round(x2s * inv_r)); y2 = int(round(y2s * inv_r))
                            all_dets_orig.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': 0.60 + 0.40*sift_conf,
                                'score_tm': 0.5,  # placeholder
                                'method': 'SIFT',
                                'template_id': t_idx
                            })

        # Multi-rotation, multi-scale TM (scaled plane)
        t_edge0 = cv2.Canny(cv2.GaussianBlur(t_s, (3,3), 0), 50, 150).astype(np.float32)/255.0
        for ang in ROTATIONS:
            M = cv2.getRotationMatrix2D((tw0/2, th0/2), ang, 1.0)
            t_rot  = cv2.warpAffine(t_s,     M, (tw0, th0), flags=cv2.INTER_LINEAR,  borderValue=255)
            te_rot = cv2.warpAffine(t_edge0, M, (tw0, th0), flags=cv2.INTER_LINEAR,  borderValue=0)

            for s in SCALES:
                tw = max(8, int(round(tw0 * s)))
                th = max(8, int(round(th0 * s)))
                if th >= Hs or tw >= Ws:
                    continue
                t_scaled  = cv2.resize(t_rot,  (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                te_scaled = cv2.resize(te_rot, (tw, th), interpolation=cv2.INTER_LINEAR)

                resp_i = cv2.matchTemplate(img_gray_eq_s, t_scaled, cv2.TM_CCOEFF_NORMED)
                resp_e = cv2.matchTemplate(img_edge_s,   te_scaled, cv2.TM_CCOEFF_NORMED)
                resp_f = (INT_W * resp_i + EDGE_W * resp_e).astype(np.float32)

                # Adaptive threshold: keep peaks close to the max; don't over-tighten
                _, maxVal, _, _ = cv2.minMaxLoc(resp_f)
                delta = 0.10 + 0.02*max(0, n_templates-1)  # 0.10..0.14
                thr_local = max(TM_THR, maxVal - delta)

                peaks = topk_peaks_from_response(resp_f, k=60, thr=thr_local,
                                                 min_dist=int(max(8, 0.25*min(th, tw))))
                for (py, px, sc) in peaks:
                    x1s, y1s = int(px), int(py)
                    x2s, y2s = x1s + tw, y1s + th
                    x1 = int(round(x1s * inv_r)); y1 = int(round(y1s * inv_r))
                    x2 = int(round(x2s * inv_r)); y2 = int(round(y2s * inv_r))
                    H0, W0 = img_gray.shape[:2]
                    if x1 >= W0 or y1 >= H0:
                        continue
                    x2 = min(W0, x2); y2 = min(H0, y2)
                    all_dets_orig.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float((1 - FUSE_W_SIFT) * sc + FUSE_W_SIFT*0.5),
                        'score_tm': float(sc),
                        'method': 'TM',
                        'template_id': t_idx
                    })

    # Global NMS
    all_dets_orig = non_max_suppression(all_dets_orig, iou_threshold=IOU_NMS) if all_dets_orig else []

    # ---------- Final verification (full-res) with cross-template margin ----------
    def verify_candidate_fullres(roi_box, my_template_bgr):
        x1, y1, x2, y2 = roi_box
        if y2 <= y1 or x2 <= x1:
            return 0.0, 0.0, 0.0
        roi_eq = img_gray_eq[y1:y2, x1:x2]
        if roi_eq.size == 0:
            return 0.0, 0.0, 0.0

        tgray = cv2.cvtColor(my_template_bgr, cv2.COLOR_BGR2GRAY)
        tgray = trim_white_border(tgray, tol=245)
        t_eq  = clahe.apply(tgray)
        h, w = roi_eq.shape[:2]
        if h < 10 or w < 10:
            return 0.0, 0.0, 0.0
        t_eq_r = cv2.resize(
            t_eq, (w, h),
            interpolation=cv2.INTER_AREA if (w*h) < (t_eq.shape[1]*t_eq.shape[0]) else cv2.INTER_LINEAR
        )

        # self scores
        resp_i = cv2.matchTemplate(roi_eq, t_eq_r, cv2.TM_CCOEFF_NORMED)
        ncc_i = float(resp_i.max()) if resp_i.size > 1 else float(resp_i[0,0])

        roi_e = edge_map(roi_eq)
        t_e   = edge_map(t_eq_r)
        resp_e = cv2.matchTemplate(roi_e, t_e, cv2.TM_CCOEFF_NORMED)
        ncc_e = float(resp_e.max()) if resp_e.size > 1 else float(resp_e[0,0])

        _, thr_r = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, thr_t = cv2.threshold(t_eq_r, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask_r = roi_eq < thr_r
        mask_t = t_eq_r < thr_t
        dsc = float(dice_coeff(mask_r, mask_t))
        hu  = float(hu_similarity(mask_r, mask_t))

        # milder penalties
        ar_roi = w / max(1.0, h)
        ar_tmp = t_eq.shape[1] / max(1.0, t_eq.shape[0])
        ar_penalty = 1.0 - min(0.25, abs(ar_roi - ar_tmp) / max(0.01, ar_tmp))
        edens = float(roi_e.mean())
        ed_penalty = 1.0 if edens >= 0.008 else 0.7

        score_self = (0.52 * ncc_i + 0.28 * ncc_e + 0.15 * dsc + 0.05 * hu) * ar_penalty * ed_penalty
        return score_self, ncc_i, ncc_e

    def best_other_ncc(roi_box, my_idx):
        x1, y1, x2, y2 = roi_box
        roi_eq = img_gray_eq[y1:y2, x1:x2]
        if roi_eq.size == 0:
            return 0.0
        h, w = roi_eq.shape[:2]
        best = 0.0
        for j, Tbgr in enumerate(templates_bgr):
            if j == my_idx:
                continue
            tgray = cv2.cvtColor(Tbgr, cv2.COLOR_BGR2GRAY)
            tgray = trim_white_border(tgray, tol=245)
            t_eq  = clahe.apply(tgray)
            t_eq_r = cv2.resize(t_eq, (w, h), interpolation=cv2.INTER_AREA if (w*h) < (t_eq.shape[1]*t_eq.shape[0]) else cv2.INTER_LINEAR)
            resp_i = cv2.matchTemplate(roi_eq, t_eq_r, cv2.TM_CCOEFF_NORMED)
            ncc_i = float(resp_i.max()) if resp_i.size > 1 else float(resp_i[0,0])
            if ncc_i > best:
                best = ncc_i
        return best

    def verify_with_settings(cands, ver_min, margin):
        out = []
        for det in cands:
            t_idx = det['template_id']
            tbgr = templates_bgr[t_idx]
            score_self, ncc_i_self, _ = verify_candidate_fullres(det['bbox'], tbgr)
            if score_self < ver_min:
                continue
            if len(templates_bgr) > 1 and margin > 0.0:
                ncc_other = best_other_ncc(det['bbox'], t_idx)
                if (ncc_i_self - ncc_other) < margin:
                    continue
            det2 = det.copy()
            det2['verify'] = score_self
            det2['confidence'] = float(0.6*score_self + 0.4*det['confidence'])
            out.append(det2)
        return out

    # First verify (user thresholds)
    verified = verify_with_settings(all_dets_orig, verify_min, margin_other)

    # Recover pass if nothing found
    if len(verified) == 0 and len(all_dets_orig) > 0:
        verified = verify_with_settings(
            all_dets_orig,
            ver_min=max(0.56, verify_min - 0.08),
            margin=0.0  # no cross-template margin in recover pass
        )

    # Per-template counts
    per_template = {}
    for t_idx in range(len(templates_bgr)):
        per_template[t_idx] = sum(1 for d in verified if d['template_id'] == t_idx)

    # Colors per template
    palette_rgb = get_palette_rgb(max(1, len(templates_bgr)))
    palette_bgr = [(c[2], c[1], c[0]) for c in palette_rgb]

    # Visualization
    vis_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for det in verified:
        x1, y1, x2, y2 = det['bbox']
        t_idx = det['template_id']
        color_bgr = palette_bgr[t_idx % len(palette_bgr)]
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), color_bgr, 3)
        label = f"{det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - th - 4)
        cv2.rectangle(vis_bgr, (x1, y_text), (x1 + tw + 6, y_text + th + 4), (255,255,255), -1)
        cv2.putText(vis_bgr, label, (x1 + 3, y_text + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
    vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    # Template strip with colored borders
    strip = draw_template_strip(templates_bgr, strip_height=120, pad=12, colors_rgb=palette_rgb)
    if strip is not None:
        strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
        gap = 8
        Hs2, Ws2 = strip_rgb.shape[:2]
        Hi, Wi = vis.shape[:2]
        canvas = np.full((Hs2 + gap + Hi, max(Ws2, Wi), 3), 255, np.uint8)
        canvas[0:Hs2, 0:Ws2] = strip_rgb
        canvas[Hs2:Hs2+gap, :] = 255
        canvas[Hs2+gap:Hs2+gap+Hi, 0:Wi] = vis
        vis = canvas

    # NOTE: do NOT save to disk here
    return {
        'detections': verified,
        'per_template': per_template,
        'total': len(verified),
        'result_image': vis,                 # in-memory RGB array
        'templates_used': len(templates_bgr),
        'template_colors_rgb': palette_rgb,  # for legend in UI
    }

# -----------------------------------
# Layout (Pro UI)
# -----------------------------------
upload_card = step_card(
    "Step 1 • Upload File",
    [
        html.P("Upload a PDF (single page) or a PNG/JPG drawing.", className="text-muted mb-2"),
        dcc.Upload(
            id='upload-file',
            children=html.Div([
                html.Span("Drag & Drop or "),
                html.A('Select file', className="fw-bold")
            ]),
            style={
                'width': '100%', 'height': '64px', 'lineHeight': '64px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
                'textAlign': 'center', 'backgroundColor': '#f8f9fa'
            },
            multiple=False
        ),
        html.Div(id='upload-status', className='mt-2')
    ]
)

crop_card = step_card(
    "Step 2 • Crop Symbol Templates",
    [
        html.P("Draw rectangles around symbols you want to detect.", className="text-muted"),
        dcc.Graph(
            id='image-graph',
            figure=create_empty_figure(),
            config={'modeBarButtonsToAdd': ['drawrect'], 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}
        ),
        html.Div(id='bbox-display', className='mt-2'),
        dbc.Row([
            dbc.Col(dbc.Button('Save Crop', id='save-crop-btn', color='success', className='w-100', n_clicks=0), width=4),
            dbc.Col(dbc.Button('Clear All Crops', id='clear-crops-btn', color='danger', className='w-100', n_clicks=0), width=4),
            dbc.Col(html.Div(id='crops-count', className='text-center pt-2 fw-semibold'), width=4),
        ], className='mt-2'),
    ]
)

controls_card = step_card(
    "Step 3 • Run Detection",
    [
        dbc.Row([
            dbc.Col([
                dbc.Label("False-Positive Guard (verify threshold)"),
                dcc.Slider(
                    id="fp-threshold",
                    min=0.55, max=0.85, step=0.01, value=0.66,
                    marks={0.55:"0.55", 0.66:"0.66", 0.75:"0.75", 0.85:"0.85"}
                ),
                html.Small("Raise to reduce FPs; lower if true matches are missed.", className="text-muted")
            ], width=12),
            dbc.Col([
                dbc.Label("Cross-Template Margin (NCC intensity)"),
                dcc.Slider(
                    id="ct-margin",
                    min=0.00, max=0.15, step=0.01, value=0.03,
                    marks={0.00:"0.00", 0.03:"0.03", 0.08:"0.08", 0.15:"0.15"}
                ),
                html.Small("A detection must beat all other templates by at least this margin.", className="text-muted")
            ], width=12, className="mt-3")
        ], className="mb-3"),
        dcc.Loading(
            id="loading-detect",
            type="dot",
            children=dbc.Button('Run Detection', id='run-detection-btn', color='primary', size='lg', className='w-100', n_clicks=0)
        ),
        html.Div(id='detection-status', className='mt-3'),
    ]
)

results_card = step_card(
    "Step 4 • View & Download Results",
    [
        dcc.Loading(id="loading-results", type="default", children=html.Div(id='results-display')),
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Button('Download Result (PDF)', id='download-pdf-btn', color='secondary',
                               className='w-100', n_clicks=0, disabled=True), width=6),
        ], className='mt-2'),
        dcc.Download(id='download-result')
    ]
)

app.layout = dbc.Container([
    app_header(),
    dbc.Row([
        dbc.Col(upload_card, md=6, lg=5),
        dbc.Col(controls_card, md=6, lg=7),
    ], className="mb-3"),
    dbc.Row([dbc.Col(crop_card, width=12)], className="mb-3"),
    dbc.Row([dbc.Col(results_card, width=12)]),
], fluid=True, style={'maxWidth': '1500px'})

# -----------------------------------
# Callbacks
# -----------------------------------
@app.callback(
    Output('image-graph', 'figure'),
    Output('upload-status', 'children'),
    Input('upload-file', 'contents'),
    State('upload-file', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return create_empty_figure(), ""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = UPLOAD_DIR / f"{timestamp}_{filename}"

        if filename.lower().endswith('.pdf'):
            with open(file_path, 'wb') as f:
                f.write(decoded)
            img_rgb = pdf_to_image(file_path, dpi=300)
        else:
            nparr = np.frombuffer(decoded, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode image.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(file_path), img)

        app_state['current_image'] = img_rgb
        app_state['current_image_path'] = str(file_path)
        app_state['img_height'], app_state['img_width'] = img_rgb.shape[:2]
        app_state['coords'] = []
        app_state['saved_crops'] = []
        app_state['crop_counter'] = 1
        app_state['detection_results'] = None

        fig = create_figure_with_image(img_rgb)
        status = dbc.Alert(
            f"Loaded: {filename}  •  {app_state['img_width']} × {app_state['img_height']}",
            color="success", className="mb-0"
        )
        return fig, status

    except Exception as e:
        return create_empty_figure(), dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output('bbox-display', 'children'),
    Input('image-graph', 'relayoutData'),
    prevent_initial_call=True
)
def capture_bbox(relayout_data):
    if app_state['current_image'] is None:
        return ""
    rect = parse_last_rect_from_relayout(relayout_data)
    if rect is None:
        return ""
    x1, y1, x2, y2 = rect
    app_state['coords'] = [{"x0": x1, "y0": y1, "x1": x2, "y1": y2}]
    return dbc.Alert(
        f"Selected: [{x1}, {y1}] → [{x2}, {y2}]  (Size: {x2 - x1} × {y2 - y1})",
        color="info", className="mb-0"
    )

@app.callback(
    Output('image-graph', 'figure', allow_duplicate=True),
    Output('crops-count', 'children'),
    Input('save-crop-btn', 'n_clicks'),
    Input('clear-crops-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_crop_actions(save_clicks, clear_clicks):
    if app_state['current_image'] is None:
        return create_empty_figure(), ""

    trigger = (callback_context.triggered[0]['prop_id'].split('.')[0]
               if callback_context.triggered else None)

    if trigger == 'clear-crops-btn':
        app_state['saved_crops'] = []
        app_state['crop_counter'] = 1
        fig = create_figure_with_image(app_state['current_image'])
        return fig, "Crops: 0"

    if trigger == 'save-crop-btn':
        if not app_state['coords']:
            fig = create_figure_with_image(app_state['current_image'], app_state['saved_crops'])
            return fig, f"Crops: {len(app_state['saved_crops'])}"

        rect = app_state['coords'][0]
        x1, y1, x2, y2 = rect["x0"], rect["y0"], rect["x1"], rect["y1"]
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            fig = create_figure_with_image(app_state['current_image'], app_state['saved_crops'])
            return fig, f"Crops: {len(app_state['saved_crops'])}"

        img_rgb = app_state['current_image']
        H, W = img_rgb.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        crop_rgb = img_rgb[y1c:y2c, x1c:x2c]
        if crop_rgb.size == 0:
            fig = create_figure_with_image(img_rgb, app_state['saved_crops'])
            return fig, f"Crops: {len(app_state['saved_crops'])}"

        filename = f"symbol_{app_state['crop_counter']:03d}.png"
        filepath = OUTPUT_DIR / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

        app_state['saved_crops'].append({
            'id': app_state['crop_counter'],
            'filename': filename,
            'filepath': str(filepath),
            'coords': [x1c, y1c, x2c, y2c],
            'size': (x2c - x1c, y2c - y1c)
        })
        app_state['crop_counter'] += 1
        app_state['coords'] = []

        fig = create_figure_with_image(app_state['current_image'], app_state['saved_crops'])
        return fig, f"Crops: {len(app_state['saved_crops'])}"

    fig = create_figure_with_image(app_state['current_image'], app_state['saved_crops'])
    return fig, f"Crops: {len(app_state['saved_crops'])}"

def _auto_save_last_rect_if_needed():
    """Auto-save last drawn rectangle as first crop if user forgot to click Save."""
    if app_state['current_image'] is None:
        return
    if app_state['coords'] and len(app_state['saved_crops']) == 0:
        rect = app_state['coords'][0]
        x1, y1, x2, y2 = rect["x0"], rect["y0"], rect["x1"], rect["y1"]
        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
            img_rgb = app_state['current_image']
            H, W = img_rgb.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x2), min(H, y2)
            crop_rgb = img_rgb[y1c:y2c, x1c:x2c]
            if crop_rgb.size > 0:
                filename = f"symbol_{app_state['crop_counter']:03d}.png"
                filepath = OUTPUT_DIR / filename
                cv2.imwrite(str(filepath), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                app_state['saved_crops'].append({
                    'id': app_state['crop_counter'],
                    'filename': filename,
                    'filepath': str(filepath),
                    'coords': [x1c, y1c, x2c, y2c],
                    'size': (x2c - x1c, y2c - y1c)
                })
                app_state['crop_counter'] += 1

@app.callback(
    Output('detection-status', 'children'),
    Output('results-display', 'children'),
    Output('download-pdf-btn', 'disabled'),
    Input('run-detection-btn', 'n_clicks'),
    State('fp-threshold', 'value'),
    State('ct-margin', 'value'),
    prevent_initial_call=True
)
def run_detection(_n, fp_thr, ct_margin):
    try:
        if app_state['current_image'] is None:
            return dbc.Alert("Please upload an image first", color="warning"), html.Div(), True

        if len(app_state['saved_crops']) == 0 and app_state['coords']:
            _auto_save_last_rect_if_needed()

        if len(app_state['saved_crops']) == 0:
            return dbc.Alert("Please save at least one symbol template (or draw a box and click Save Crop).",
                             color="warning"), html.Div(), True

        template_paths = [c['filepath'] for c in app_state['saved_crops']]
        results = run_symbol_detection(
            app_state['current_image_path'],
            template_paths,
            verify_min=float(fp_thr),
            margin_other=float(ct_margin)
        )

        if results is None:
            # show prompts anyway
            thumbs = []
            for c in app_state['saved_crops']:
                bgr = cv2.imread(c['filepath'])
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                thumbs.append(
                    px.imshow(rgb).update_layout(
                        height=180,
                        margin=dict(l=0, r=0, t=30, b=0),
                        title=c['filename']
                    )
                )
            prompt_row = html.Div([dcc.Graph(figure=f) for f in thumbs]) if thumbs else html.Div("No templates loaded.")
            return dbc.Alert("Detection failed. Showing your visual prompts.", color="danger"), prompt_row, True

        app_state['detection_results'] = results

        # Legend
        legend_items = []
        colors = results.get('template_colors_rgb', [])
        for idx, c in enumerate(colors):
            name = app_state['saved_crops'][idx]['filename'] if idx < len(app_state['saved_crops']) else f"template_{idx}"
            legend_items.append(
                html.Div([
                    html.Span(style={
                        "display": "inline-block", "width": "14px", "height": "14px",
                        "backgroundColor": f"rgb({c[0]},{c[1]},{c[2]})",
                        "borderRadius": "3px", "marginRight": "8px", "border": "1px solid rgba(0,0,0,0.2)"
                    }),
                    html.Span(name)
                ], style={"marginRight":"16px", "marginBottom":"6px", "display":"inline-flex", "alignItems":"center"})
            )

        status = dbc.Alert(
            f"Detection Complete! Found {results['total']} symbols across {results['templates_used']} templates "
            f"(Verify ≥ {fp_thr:.2f}, margin ≥ {ct_margin:.2f}). Result is in memory; use 'Download Result (PDF)' to save.",
            color=("success" if results['total'] > 0 else "warning"), className="mb-2"
        )

        fig_result = px.imshow(results['result_image'])
        fig_result.update_layout(height=900, title=f"Detection Results ({results['total']} symbols)",
                                 margin=dict(l=10, r=10, t=50, b=10))

        per_t_items = []
        for idx, cnt in results['per_template'].items():
            c = colors[idx % len(colors)] if colors else (0,0,0)
            name = app_state['saved_crops'][idx]['filename'] if idx < len(app_state['saved_crops']) else f"template_{idx}"
            per_t_items.append(
                html.Li([
                    html.Span(style={
                        "display":"inline-block","width":"12px","height":"12px",
                        "backgroundColor": f"rgb({c[0]},{c[1]},{c[2]})",
                        "borderRadius":"2px","marginRight":"8px","border":"1px solid rgba(0,0,0,0.2)"
                    }),
                    f"{name}: {cnt}"
                ])
            )

        results_display = html.Div([
            html.Div(legend_items, style={"marginBottom":"6px"}),
            dcc.Graph(figure=fig_result),
            html.Hr(),
            html.H5("Per-template counts"),
            html.Ul(per_t_items),
            html.Hr(),
            html.H6("Details (in-memory)", className="text-muted"),
            html.Ul([
                html.Li(f"Total Detections: {results['total']}"),
                html.Li(f"Templates Used: {results['templates_used']}"),
                html.Li("Result is NOT saved automatically; click 'Download Result (PDF)' to save."),
                html.Li("Detector: SIFT/FLANN + multi-scale/rotation TM + global NMS"),
                html.Li("Guards: adaptive TM threshold, cross-template margin, full-res verify (NCC+edge+Dice+Hu)"),
                html.Li("Fallback: 'recover' pass lowers verify threshold & removes margin if zero hits"),
            ], className="small")
        ])
        return status, results_display, False

    except Exception as e:
        return dbc.Alert(f"Error during detection: {e}", color="danger"), html.Div(), True

# Download PDF (in-memory; no disk write until clicked)
@app.callback(
    Output('download-result', 'data'),
    Input('download-pdf-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_result_pdf(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    if app_state['detection_results'] is None:
        raise PreventUpdate
    try:
        arr = app_state['detection_results']['result_image']
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_result_{ts}.pdf"
        return dcc.send_bytes(lambda buff: img.save(buff, "PDF", resolution=300.0), filename)
    except Exception:
        raise PreventUpdate

# -----------------------------------
# Main
# -----------------------------------
if __name__ == '__main__':
    print("\n" + "="*92)
    print("SYMBOL DETECTION PRO — Enhanced v4.3 (Pro UI + Softer Defaults + Recover Pass, NO auto-save)")
    print("="*92)
    print("Tips:")
    print(" • If multi-template still misses: lower 'Verify threshold' to ~0.62 and/or reduce 'Margin' to 0.01.")
    print(" • If you see new FPs: raise 'Verify threshold' to 0.70–0.75 or increase 'Margin' to 0.06–0.10.")
    print("Open: http://127.0.0.1:8050")
    print("="*92 + "\n")
    app.run(debug=False, host='127.0.0.1', port=4000)
