"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    c = cfg["label_format"]
    issues = []
    for lf in ds.label_files():
        stem = lf.stem
        size = ds.image_size(stem)                 # None nếu ảnh không mở được: bỏ check BOX_TINY
        kept = []                                  # (lớp, box) của các dòng hợp lệ ĐỨNG TRƯỚC, để so DUP_BOX
        for line, text in ds.read_label_lines(stem) or []:
            try:
                cls, x, y, w, h = parse_yolo_line(text)
            except ValueError as e:                # dòng thiếu cột / nan / không phải số: không được làm script chết
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, f"dòng sai định dạng: {e}", line,
                                    {"text": text.strip()[:60]}))
                continue
            if not 0 <= cls < len(ds.classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem,
                                    f"id lớp {cls} ngoài 0..{len(ds.classes) - 1}", line, {"class_id": cls}))
                continue
            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem,
                                    f"box rộng/cao bằng 0 (w={w}, h={h})", line))
                continue

            # --- hình học: chỉ chạy khi dòng đã hợp lệ về cú pháp và lớp
            overflow = max(-(x - w / 2), -(y - h / 2), (x + w / 2) - 1.0, (y + h / 2) - 1.0)
            if overflow > c["overflow_tol"]:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                                    f"box tràn ảnh {overflow:.4f} > dung sai {c['overflow_tol']}", line,
                                    {"overflow": round(overflow, 6)}))
            dup = next((i for i, (kc, kb) in enumerate(kept, 1)
                        if kc == cls and iou_xywh((x, y, w, h), kb) > c["dup_box_iou"]), None)
            if dup is not None:
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem,
                                    f"lặp box cùng lớp {cls} với một dòng trước đó", line, {"class_id": cls}))
            if size:
                short_px = min(w * size[0], h * size[1])
                if short_px < c["min_box_px"]:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem,
                                        f"cạnh ngắn {short_px:.1f}px < {c['min_box_px']}px", line,
                                        {"short_px": round(short_px, 1)}))
            if w * h > c["max_box_frac"]:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem,
                                    f"box chiếm {w * h:.2%} ảnh > {c['max_box_frac']:.0%}", line,
                                    {"area_frac": round(w * h, 4)}))
            kept.append((cls, (x, y, w, h)))
    return issues
