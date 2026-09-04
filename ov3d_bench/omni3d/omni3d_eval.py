# Copied and adapted from OVM3D-Det/cubercnn/evaluation/omni3d_evaluation.py
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
import datetime
import time
from collections import defaultdict

import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
from pytorch3d import _C
from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles
try:
    from mmdet3d.ops.iou3d import boxes_iou3d as _boxes_iou3d
except Exception:
    _boxes_iou3d = None


MAX_DTS_CROSS_GTS_FOR_IOU3D = 0


def _check_coplanar(boxes, eps=1e-4):
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    b = boxes.shape[0]
    p, v = faces.shape
    v0, v1, v2, v3 = verts.reshape(b, p, v, 3).unbind(2)

    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    mat1 = (v3 - v0).view(b, 1, -1)
    mat2 = normal.view(b, -1, 1)

    return (mat1.bmm(mat2).abs() < eps).view(b)


def _check_nonzero(boxes, eps=1e-8):
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    b = boxes.shape[0]
    t, v = faces.shape
    v0, v1, v2 = verts.reshape(b, t, v, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_areas = normals.norm(dim=-1) / 2
    return (face_areas > eps).all(1).view(b)


def _corners_to_7d_upright(boxes):
    centers = boxes.mean(dim=1)
    xy = boxes[:, :, :2] - centers[:, None, :2]

    cov = torch.bmm(xy.transpose(1, 2), xy) / float(xy.shape[1])
    eigvals, eigvecs = torch.linalg.eigh(cov)
    principal = eigvecs[:, :, 1]
    yaw = torch.atan2(principal[:, 1], principal[:, 0])

    cos = torch.cos(-yaw)
    sin = torch.sin(-yaw)
    rot = torch.stack(
        [
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin, cos], dim=1),
        ],
        dim=1,
    )
    xy_rot = torch.bmm(xy, rot)

    x_min, _ = xy_rot[:, :, 0].min(dim=1)
    x_max, _ = xy_rot[:, :, 0].max(dim=1)
    y_min, _ = xy_rot[:, :, 1].min(dim=1)
    y_max, _ = xy_rot[:, :, 1].max(dim=1)

    z_min, _ = boxes[:, :, 2].min(dim=1)
    z_max, _ = boxes[:, :, 2].max(dim=1)

    width = x_max - x_min
    length = y_max - y_min
    height = z_max - z_min
    center_z = (z_min + z_max) * 0.5

    return torch.stack(
        [
            centers[:, 0],
            centers[:, 1],
            center_z,
            width,
            length,
            height,
            yaw,
        ],
        dim=1,
    )


def _iou3d_upright(boxes_dt, boxes_gt):
    if _boxes_iou3d is None:
        raise RuntimeError(
            "mmdet3d is required for GPU 3D IoU (install mmdet3d with iou3d ops)"
        )
    dt_7d = _corners_to_7d_upright(boxes_dt)
    gt_7d = _corners_to_7d_upright(boxes_gt)
    return _boxes_iou3d(dt_7d, gt_7d)


def box3d_overlap(
    boxes_dt,
    boxes_gt,
    eps_coplanar=1e-4,
    eps_nonzero=1e-8,
    backend="pytorch3d",
):
    if boxes_dt.numel() == 0 or boxes_gt.numel() == 0:
        return torch.zeros((boxes_dt.shape[0], boxes_gt.shape[0]), device=boxes_dt.device)

    if not torch.isfinite(boxes_dt).all() or not torch.isfinite(boxes_gt).all():
        print("Input boxes contain non-finite values")
        return torch.zeros((boxes_dt.shape[0], boxes_gt.shape[0]), device=boxes_dt.device)

    invalid_coplanar = ~_check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero = ~_check_nonzero(boxes_dt, eps=eps_nonzero)

    if backend == "mmdet3d":
        ious = _iou3d_upright(boxes_dt, boxes_gt)
    elif backend == "pytorch3d":
        ious = _C.iou_box3d(boxes_dt, boxes_gt)[1]
    else:
        raise ValueError("Unknown 3D IoU backend: {}".format(backend))

    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0

    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0

    return ious


class Omni3DParams:
    def setDet2DParams(self):
        self.imgIds = []
        self.catIds = []

        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]


    def setDet3DParams(self):
        self.imgIds = []
        self.catIds = []

        self.iouThrs = np.linspace(
            0.05, 0.5, int(np.round((0.5 - 0.05) / 0.05)) + 1, endpoint=True
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [[0, 1e5], [0, 10], [10, 35], [35, 1e5]]

    def __init__(self, mode="2D"):
        if mode == "2D":
            self.setDet2DParams()
        elif mode == "3D":
            self.setDet3DParams()
        else:
            raise Exception("mode %s not supported" % (mode))

        self.iouType = "bbox"
        self.mode = mode
        self.proximity_thresh = 0.3


class Omni3Deval:
    def __init__(self, cocoGt=None, cocoDt=None, iouType="bbox", mode="2D", eval_prox=False):
        if not iouType:
            print("iouType not specified. use default iouType bbox")
        elif iouType != "bbox":
            print("no support for %s iouType" % (iouType))
        self.mode = mode
        if mode not in ["2D", "3D"]:
            raise Exception("mode %s not supported" % (mode))
        self.eval_prox = eval_prox
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt

        self.evalImgs = defaultdict(list)

        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.params = Omni3DParams(mode)
        self._paramsEval = {}
        self.stats = []
        self.ious = {}

        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.evals_per_cat_area = None

    def _prepare(self):
        p = self.params

        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

        ignore_flag = "ignore2D" if self.mode == "2D" else "ignore3D"
        for gt in gts:
            gt[ignore_flag] = gt[ignore_flag] if ignore_flag in gt else 0

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(list)
        self.eval = {}

    def accumulate(self, p=None):
        print("Accumulating evaluation results...")
        assert self.evalImgs, "Please run evaluate() first"

        tic = time.time()

        if p is None:
            p = self.params

        p.catIds = p.catIds

        t_count = len(p.iouThrs)
        r_count = len(p.recThrs)
        k_count = len(p.catIds)
        a_count = len(p.areaRng)
        m_count = len(p.maxDets)

        precision = -np.ones((t_count, r_count, k_count, a_count, m_count))
        recall = -np.ones((t_count, k_count, a_count, m_count))
        scores = -np.ones((t_count, r_count, k_count, a_count, m_count))

        _pe = self._paramsEval

        catIds = _pe.catIds
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)

        catid_list = [k for n, k in enumerate(p.catIds) if k in setK]
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]

        i0_count = len(_pe.imgIds)
        a0_count = len(_pe.areaRng)

        has_precomputed_evals = not (self.evals_per_cat_area is None)

        if has_precomputed_evals:
            evals_per_cat_area = self.evals_per_cat_area
        else:
            evals_per_cat_area = {}

        for k, (k0, catId) in enumerate(zip(k_list, catid_list)):
            nk = k0 * a0_count * i0_count
            for a, a0 in enumerate(a_list):
                na = a0 * i0_count

                if has_precomputed_evals:
                    evals = evals_per_cat_area[(catId, a)]
                else:
                    evals = [self.evalImgs[nk + na + i] for i in i_list]
                    evals = [e for e in evals if not e is None]
                    evals_per_cat_area[(catId, a)] = evals

                if len(evals) == 0:
                    continue

                for m, maxDet in enumerate(m_list):
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in evals])
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in evals], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in evals], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in evals])
                    npig = np.count_nonzero(gtIg == 0)

                    if npig == 0:
                        continue

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((r_count,))
                        ss = np.zeros((r_count,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")

                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except Exception:
                            pass

                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

        self.evals_per_cat_area = evals_per_cat_area

        self.eval = {
            "params": p,
            "counts": [t_count, r_count, k_count, a_count, m_count],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }

        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def evaluate(self):
        print("Running per image evaluation...")

        p = self.params
        print("Evaluate annotation type *{}*".format(p.iouType))
        tic = time.time()

        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))

        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()

        catIds = p.catIds

        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]

        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]

        self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, imgId, catId):
        device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) == 0:
            return []

        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "bbox":
            if self.mode == "2D":
                g = [g["bbox"] for g in gt]
                d = [d["bbox"] for d in dt]
            elif self.mode == "3D":
                g = [g["bbox3D"] for g in gt]
                d = [d["bbox3D"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        iscrowd = [0 for o in gt]
        if self.mode == "2D":
            ious = maskUtils.iou(d, g, iscrowd)
        elif len(d) > 0 and len(g) > 0:
            if torch.cuda.is_available() and len(d) * len(g) < MAX_DTS_CROSS_GTS_FOR_IOU3D:
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

            dd = torch.tensor(d, device=device, dtype=torch.float32)
            gg = torch.tensor(g, device=device, dtype=torch.float32)

            ious = box3d_overlap(dd, gg).cpu().numpy()
        else:
            ious = []

        in_prox = None

        if self.eval_prox:
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
            iscrowd = [0 for o in gt]
            ious2d = maskUtils.iou(d, g, iscrowd)

            if type(ious2d) == list:
                in_prox = []
            else:
                in_prox = ious2d > p.proximity_thresh

        return ious, in_prox

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) == 0:
            return None

        flag_range = "area" if self.mode == "2D" else "depth"
        flag_ignore = "ignore2D" if self.mode == "2D" else "ignore3D"

        for g in gt:
            if g[flag_ignore] or (g[flag_range] < aRng[0] or g[flag_range] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]

        ious = (
            self.ious[imgId, catId][0][:, gtind]
            if len(self.ious[imgId, catId][0]) > 0
            else self.ious[imgId, catId][0]
        )

        if self.eval_prox:
            in_prox = (
                self.ious[imgId, catId][1][:, gtind]
                if len(self.ious[imgId, catId][1]) > 0
                else self.ious[imgId, catId][1]
            )

        t_count = len(p.iouThrs)
        g_count = len(gt)
        d_count = len(dt)
        gtm = np.zeros((t_count, g_count))
        dtm = np.zeros((t_count, d_count))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((t_count, d_count))

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1

                    for gind, g in enumerate(gt):
                        if self.eval_prox and not in_prox[dind, gind]:
                            continue

                        if gtm[tind, gind] > 0:
                            continue

                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break

                        if ious[dind, gind] < iou:
                            continue

                        iou = ious[dind, gind]
                        m = gind

                    if m == -1:
                        continue

                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

        a = np.array(
            [d[flag_range] < aRng[0] or d[flag_range] > aRng[1] for d in dt]
        ).reshape((1, len(dt)))

        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, t_count, 0)))

        if self.eval_prox and len(in_prox) > 0:
            dt_far = in_prox.any(1) == 0
            dtIg = np.logical_or(dtIg, np.repeat(dt_far.reshape((1, len(dt))), t_count, 0))

        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def summarize(self):
        def _summarize(mode, ap=1, iouThr=None, maxDets=100, log_str=""):
            p = self.params
            eval = self.eval

            if mode == "2D":
                iStr = (" {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}")
            elif mode == "3D":
                iStr = " {:<18} {} @[ IoU={:<9} | depth={:>6s} | maxDets={:>3d} ] = {:0.3f}"

            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"

            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [0]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            area_label = "all"

            if ap == 1:
                s = eval["precision"]

                if iouThr is not None:
                    t = np.where(np.isclose(iouThr, p.iouThrs.astype(float)))[0]
                    s = s[t]

                s = s[:, :, :, aind, mind]
            else:
                s = eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]

            if ap == 1 and iouThr is None and mode == "3D":
                ap_per_thr = []
                for tind in range(s.shape[0]):
                    s_t = s[tind]
                    if len(s_t[s_t > -1]) == 0:
                        ap_per_thr.append(-1)
                    else:
                        ap_per_thr.append(np.mean(s_t[s_t > -1]))
                valid_ap = [v for v in ap_per_thr if v >= 0]
                mean_s = -1 if len(valid_ap) == 0 else float(np.mean(valid_ap))
            else:
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])

            if log_str != "":
                log_str += "\n"

            log_str += "mode={} ".format(mode) + \
                iStr.format(titleStr, typeStr, iouStr, area_label, maxDets, mean_s)

            return mean_s, log_str

        def _summarizeDets(mode):
            params = self.params

            thres = [0.5, 0.75, 0.95] if mode == "2D" else [0.15, 0.25, 0.5]

            stats = np.zeros((7,))
            stats[0], log_str = _summarize(mode, 1)

            stats[1], log_str = _summarize(
                mode, 1, iouThr=thres[0], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[2], log_str = _summarize(
                mode, 1, iouThr=thres[1], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[3], log_str = _summarize(
                mode, 1, iouThr=thres[2], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[4], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[0], log_str=log_str
            )

            stats[5], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[1], log_str=log_str
            )

            stats[6], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[2], log_str=log_str
            )

            return stats, log_str

        if not self.eval:
            raise Exception("Please run accumulate() first")

        stats, log_str = _summarizeDets(self.mode)
        self.stats = stats

        return log_str
