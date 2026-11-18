"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}        # dòng hỏng cú pháp -> xoá dòng
CLIP_CODES = {"BOX_OUT_OF_IMAGE"}                                        # box tràn -> cắt về [0,1]
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}                # + mọi mã DUP_IMAGE_*


def _fmt(cls, box):
    return " ".join([str(int(cls))] + [f"{v:.6f}" for v in box])


def _clip(box):
    """(cx, cy, w, h) -> box đã cắt vào trong [0,1]; None nếu cắt xong không còn diện tích."""
    x, y, w, h = box
    x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
    x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def fix(ds, issues, cfg):
    errs = [i for i in issues if i.severity == ERROR]          # cảnh báo là việc của người: không chạm
    fixes = []

    # --- 1) ảnh phải loại khỏi lô (trùng, rò rỉ test, hỏng, quá nhỏ): xoá ảnh + nhãn
    removed = set()
    for img in sorted({i.image for i in errs
                       if i.code in REMOVE_CODES or i.code.startswith("DUP_IMAGE_")}):
        ip = ds.image_path(img)
        if ip is None and not ds.label_path(img).exists():
            continue                                           # đã bị xoá ở lần chạy trước: im lặng
        name = ip.name if ip else f"{img}.txt"
        if ip:
            ip.unlink()
        ds.label_path(img).unlink(missing_ok=True)
        ds._size_cache.pop(img, None)
        removed.add(img)
        fixes.append(dict(image=img, line="", action="remove_image", before=name, after="(đã xoá khỏi lô)"))

    # --- 2) tên lớp trong data.yaml: ghi lại đúng danh sách dự án (một lần, dù có nhiều dòng lỗi)
    if any(i.code == "BAD_CLASS_NAME" for i in errs):
        before = dict(sorted(ds.yaml_names().items()))
        d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) if ds.yaml_path.exists() else {}
        d = d if isinstance(d, dict) else {}
        after = dict(enumerate(cfg["classes"]))
        if before != after:
            d["names"] = after
            ds.yaml_path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append(dict(image="data.yaml", line="", action="rename_classes",
                              before=str(before), after=str(after)))

    # --- 3) lỗi theo DÒNG: gom theo file rồi ghi lại MỘT lần (số dòng trong issues là số dòng GỐC)
    drop, clip = defaultdict(dict), defaultdict(dict)
    for i in errs:
        if i.image in removed or i.line is None:
            continue
        if i.code in DROP_CODES:
            drop[i.image][i.line] = i.code
        elif i.code in CLIP_CODES:
            clip[i.image][i.line] = i.code
    for stem in sorted(set(drop) | set(clip)):
        lp = ds.label_path(stem)
        if not lp.exists():
            continue
        raw = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        out, per_file = [], []
        for n, text in enumerate(raw, 1):                      # dòng trống vẫn ĐẾM vào số dòng gốc
            if not text.strip():
                continue
            if n in drop[stem]:                                # xoá dòng THẮNG cắt box: chỉ ghi 1 fix
                per_file.append(dict(image=stem, line=n, action="drop_line",
                                     before=text.strip(), after="(xoá dòng)"))
                continue
            if n in clip[stem]:
                try:
                    cls, *box = parse_yolo_line(text)
                except ValueError:
                    continue                                   # dòng vừa hỏng vừa tràn: xoá luôn cho an toàn
                new = _clip(tuple(box))
                if new is None:                                # box nằm hẳn ngoài ảnh: xoá, không ghi box rỗng
                    per_file.append(dict(image=stem, line=n, action="drop_line",
                                         before=text.strip(), after="(xoá dòng: box ngoài ảnh)"))
                    continue
                line_new = _fmt(cls, new)
                per_file.append(dict(image=stem, line=n, action="clip_box",
                                     before=text.strip(), after=line_new))
                out.append(line_new)
                continue
            out.append(text.strip())
        if per_file:
            lp.write_text("".join(l + "\n" for l in out), encoding="utf-8")
            fixes.extend(per_file)

    # --- 4) ảnh chưa có file nhãn: máy KHÔNG bịa nhãn, cũng không tạo file rỗng -> chuyển người gán
    for img in sorted({i.image for i in errs if i.code == "MISSING_LABEL_FILE"} - removed):
        if ds.label_path(img).exists():
            continue
        fixes.append(dict(image=img, line="", action="needs_label",
                          before="(không có file)", after="chuyển người gán"))
    return fixes
