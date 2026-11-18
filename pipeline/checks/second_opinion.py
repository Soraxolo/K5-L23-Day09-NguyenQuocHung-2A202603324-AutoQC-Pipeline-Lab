"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def _labels_with_lines(ds):
    """[(số dòng gốc, lớp, box)] cho các dòng parse được và id lớp hợp lệ — giữ số dòng để báo FALSE_POS."""
    def _for(stem):
        out = []
        for line, text in ds.read_label_lines(stem) or []:
            try:
                cls, *box = parse_yolo_line(text)
            except ValueError:
                continue                                  # dòng hỏng: label_format đã báo, ở đây bỏ qua êm
            if 0 <= cls < len(ds.classes):
                out.append((line, cls, tuple(box)))
        return out
    return _for


def run(ds, cfg, preds=None):
    c = cfg["second_opinion"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    labels_of = _labels_with_lines(ds)
    issues = []

    for p in ds.image_files():
        stem = p.stem
        rows = preds.get(stem)
        size = ds.image_size(stem)
        if not rows or ds.read_label_lines(stem) is None or not size:
            continue                                      # không có dự đoán / chưa có file nhãn / ảnh hỏng
        W, H = size
        labs = labels_of(stem)

        # --- dự đoán đáng tin: đủ conf và đủ to (guideline: vật quá nhỏ thì không gán)
        trusted = [r for r in rows if r[1] >= c["conf"] and (r[4] * W) * (r[5] * H) >= c["min_area_px"]]
        for cls_m, conf, *pbox in trusted:
            pbox = tuple(pbox)
            cands = [(iou_xywh(pbox, b), lc) for _, lc, b in labs
                     if iou_xywh(pbox, b) >= c["iou_missing"]]
            same = [i for i, lc in cands if lc == cls_m]
            if not cands:                                 # model thấy vật, nhãn không có box nào ở đó
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_MISSING", stem,
                                    f"model thấy {ds.classes[cls_m]} (conf {conf:.2f}) mà nhãn không có box",
                                    extra={"conf": round(float(conf), 4), "box": [round(v, 4) for v in pbox]}))
            elif same:
                best = max(same)
                if best < c["iou_loose"]:                 # đúng vật, đúng lớp, nhưng box lệch/lỏng
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                                        f"box {ds.classes[cls_m]} lệch so với model: IoU {best:.2f} < {c['iou_loose']}",
                                        extra={"conf": round(float(conf), 4), "iou": round(best, 4)}))
                # IoU đủ cao: nhãn đúng lớp đã phủ vật này -> im lặng
            else:
                best_other = max(i for i, _ in cands)
                if best_other >= c["iou_match"]:          # cùng vật nhưng nhãn ghi lớp khác
                    other = next(lc for i, lc in cands if i == best_other)
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                                        f"nhãn ghi {ds.classes[other]}, model nói {ds.classes[cls_m]} "
                                        f"(conf {conf:.2f}, IoU {best_other:.2f})",
                                        extra={"conf": round(float(conf), 4), "iou": round(best_other, 4),
                                               "label_class": ds.classes[other], "model_class": ds.classes[cls_m]}))
                # chỉ khác lớp với IoU mơ hồ [iou_missing, iou_match): không cờ, chờ lần duyệt khác

        # --- box nhãn không khớp vật nào model thấy (không lọc diện tích, ngưỡng conf riêng fp_conf)
        seen = [r for r in rows if r[1] >= c["fp_conf"]]
        for line, _, b in labs:
            ious = [iou_xywh(b, tuple(r[2:])) for r in seen]
            if not ious or max(ious) < c["iou_missing"]:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                                    "box nhãn không khớp dự đoán nào của model", line,
                                    extra={"iou": round(max(ious), 4) if ious else 0.0}))
    return issues
