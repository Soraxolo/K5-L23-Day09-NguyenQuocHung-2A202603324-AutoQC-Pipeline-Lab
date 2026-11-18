"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    stems = set(image_stems)
    reviewed = set(reviewed)
    thr = cfg["gate"]["warn_ratio_hold"]

    # Lỗi bắt buộc: đếm HẾT, kể cả ở ảnh người đã duyệt và lỗi cấp dataset (data.yaml) — không ai được bỏ qua.
    errors = [i for i in issues if i.severity == ERROR]
    # Cảnh báo: chỉ tính theo ẢNH còn trong lô và CHƯA có người xem.
    warn_images = {i.image for i in issues
                   if i.severity == WARN and i.image in stems and i.image not in reviewed}

    n_images = len(stems)
    warn_ratio = round(len(warn_images) / n_images, 4) if n_images else 0.0

    reasons = []
    if errors:
        decision = "Rework"
        top = sorted({i.code for i in errors})[:4]
        reasons.append(f"còn {len(errors)} lỗi bắt buộc trên {len({i.image for i in errors})} mục "
                       f"({', '.join(top)}{'...' if len({i.code for i in errors}) > 4 else ''}): trả về bước gán nhãn, KHÔNG xuất dữ liệu")
    elif warn_ratio >= thr:
        decision = "Hold"
        reasons.append(f"hết lỗi bắt buộc, nhưng {len(warn_images)}/{n_images} ảnh ({warn_ratio:.1%}) "
                       f"còn cảnh báo chưa ai duyệt ≥ ngưỡng {thr:.0%}: giữ lại, cần thêm ngân sách duyệt")
    else:
        decision = "Release"
        reasons.append(f"không còn lỗi bắt buộc; {len(warn_images)}/{n_images} ảnh ({warn_ratio:.1%}) "
                       f"còn cảnh báo chưa duyệt < ngưỡng {thr:.0%}")
    if reviewed:
        reasons.append(f"{len(reviewed)} ảnh đã qua mắt người duyệt")

    return dict(decision=decision, n_errors=len(errors), n_warn_images=len(warn_images),
                n_images=n_images, warn_ratio=warn_ratio, reasons=reasons)
