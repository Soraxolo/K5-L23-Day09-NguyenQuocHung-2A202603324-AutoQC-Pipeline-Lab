# Báo cáo Day 09 — Nguyễn Quốc Hưng, 2A202603324

## 1. Kết luận trước

Pipeline kết luận **Hold** (mã thoát 10). 224 ảnh vào, **202 ảnh ra** bản xuất `submission/export/clean_v1/`
— nhưng đây là **bản nháp để thí nghiệm, chưa phải bản phát hành**: manifest ghi `decision: Hold`.

Lý do: sau sửa tự động và 80 ảnh người duyệt, lô **không còn lỗi bắt buộc nào** (`07_error_list_v3`: 0 ERROR),
nhưng vẫn còn **38/202 ảnh (18,8%) mang cảnh báo mà chưa ai xem** — trên ngưỡng `warn_ratio_hold = 10%`.
Hàng đợi duyệt có 125 ảnh trong khi ngân sách chỉ 80: phần dư không ai gánh, nên cổng giữ lô lại.

## 2. Lỗi và phân việc

Từ `02_error_list_v1` (quét 224 ảnh ở cửa vào, 202 ảnh ở vòng QC). Đơn vị: mã cấp ảnh đếm **ảnh**, mã cấp dòng đếm **dòng nhãn**.

| Mã lỗi | Số ảnh/dòng | Ai sửa | Vì sao |
|---|---:|---|---|
| `DUP_IMAGE_EXACT` | 8 ảnh | máy | md5 trùng từng byte — quy tắc tuyệt đối, không cần người |
| `LEAK_TEST` | 5 ảnh | máy | trùng ảnh kho test/val: để lại là tự thổi phồng điểm đo |
| `DUP_IMAGE_NEAR` | 4 ảnh | máy | phash ≤ 12, đã mở xem tay cả 17 cặp (mục 3) |
| `BOX_OUT_OF_IMAGE` | 8 dòng | máy | cắt về `[0,1]` là phép toán xác định, không cần hiểu ảnh |
| `DUP_BOX` | 7 dòng | máy | cùng lớp, IoU > 0,9 với dòng trước — xoá dòng sau |
| `BAD_CLASS_ID` | 6 dòng | máy | id 93 / 7 / −1 / 4 nằm ngoài 0..3, không suy ra được lớp đúng ⇒ xoá dòng |
| `BOX_ZERO` | 4 dòng | máy | w hoặc h = 0: box vô nghĩa |
| `BAD_FORMAT` | 3 dòng | máy | thiếu cột / `nan` |
| `BAD_CLASS_NAME` | 2 dòng yaml | máy | `data.yaml` ghi sai tên lớp — ghi đè bằng `cfg["classes"]` |
| `CORRUPT_IMAGE` | 2 ảnh | máy | `Image.load()` ném lỗi (ảnh cắt cụt) |
| `BAD_SIZE` | 3 ảnh | máy | cạnh ngắn < 250px |
| **`MISSING_LABEL_FILE`** | **6 ảnh** | **người** | máy **không bịa nhãn**: chỉ ghi `needs_label`, đẩy lên đầu hàng đợi |
| `POSSIBLE_MISSING` | 53 dòng | người | model thấy vật mà nhãn không có — model cũng sai được |
| `POSSIBLE_FALSE_POS` | 29 dòng | người | box nhãn không khớp dự đoán nào |
| `POSSIBLE_LOOSE_BOX` | 28 dòng | người | box đúng lớp nhưng lệch |
| `POSSIBLE_WRONG_CLASS` | 18 dòng | người | car↔truck↔bus: đúng/sai là chuyện **ý nghĩa**, không phải quy tắc |
| `EMPTY_LABEL` | 12 ảnh | người | "ảnh nền" hay "chưa gán"? máy không phân biệt được |
| `BOX_TINY` / `BOX_HUGE` | 6 / 3 dòng | người | quy tắc dự án, cần người xác nhận vật có thật nhỏ/to không |
| `CLASS_RARE` | 4 (cấp lô) | người | lớp hiếm là tín hiệu phân bố, không phải lỗi của một ảnh |

Máy sửa **35 chỗ** (`03_fixes.csv`: 20 `drop_line`, 8 `clip_box`, 1 `rename_classes`, 6 `needs_label`), sau đó
`04_error_list_v2` chỉ còn **6 lỗi bắt buộc** — đúng 6 ảnh thiếu file nhãn mà máy từ chối bịa.

**Hàng đợi xếp theo trọng số lỗi × độ tin của model.** Đầu hàng là `img_0020` (11,52: vừa bị xoá dòng, vừa
`POSSIBLE_WRONG_CLASS` trọng số 3 — nhãn ghi *truck* cho chiếc **bus** Metro, model chắc 0,95), rồi 6 ảnh `NEEDS_LABEL`
(trọng số 10) vì không có người thì những ảnh đó vĩnh viễn kẹt ở Rework.
Với ngân sách 80 ảnh, người xem được **94/123 dòng cảnh báo ngữ nghĩa (76%)**, và cả 6 ảnh `NEEDS_LABEL` đều được gán.
Chỗ cắt rất sát: ảnh thứ 80 có ưu tiên 3,343, ảnh thứ 81 là 3,333 — thực chất là hoà.
45 ảnh bị bỏ lại chủ yếu là `POSSIBLE_LOOSE_BOX` (16) và `POSSIBLE_FALSE_POS` (11) — đúng hai mã có precision thấp nhất
trên calib (0,20 và 0,36), nghĩa là phần dư của hàng đợi cũng là phần **nhiều báo động giả nhất**.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall trên calib (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | `DUP_IMAGE_NEAR` 1,00 / 0,50 → **1,00 / 0,67**; `DUP_IMAGE_EXACT` 1,00 / 1,00 (không đổi) | Cờ này sinh lỗi **bắt buộc và xoá ảnh**, nên precision là thứ không được đánh đổi. Calib cho precision 1,00 ở mọi giá trị 2–16 nên **không phân biệt được** trong khoảng đó; recall nhảy 0,50 → 0,67 tại 4 rồi đứng yên tới 14. Tôi phá thế hoà bằng dữ liệu raw: tại 12 có đúng 17 ảnh bị loại và tôi **mở xem từng cặp** — cả 17 đều là cùng một cảnh (cặp xa nhất `img_0186`~`img_0061`, phash 10, là ảnh ghế băng công viên chụp lệch sáng). Lấy 12 để ôm trọn cặp phash 10 mà chưa chạm vùng hỏng. |
| `second_opinion.conf` | 0,25 | **0,6** | `POSSIBLE_MISSING` 0,50 / 1,00 → **0,57 / 1,00**; `POSSIBLE_WRONG_CLASS` 0,56 / 0,83 → **0,62 / 0,83**; `POSSIBLE_FALSE_POS` 0,36 / 1,00 (không đổi) | Cờ này chỉ **cảnh báo cho người**, nên về nguyên tắc ưu tiên recall. Nhưng recall ở đây không mất gì: 0,6 là giá trị **cao nhất còn giữ recall `MISSING` = 1,00** (sang 0,7 tụt còn 0,88, sang 0,8 còn 0,75), đồng thời precision của hai mã có mẫu tốt nhất đều tăng. Vì ngân sách duyệt chỉ 80 ảnh, mỗi báo động giả **chiếm chỗ** của một lỗi thật trong hàng đợi — precision ở đây là tiền thật, không chỉ là công xem. Hàng đợi co từ 147 (conf 0,1) xuống 125 ảnh. |

**Một chỗ calibration suýt đánh lừa tôi.** Cột precision của `phash_dist` trên calib là **1,00 ở cả giá trị 16 và 20**,
còn recall lại *tăng* lên 0,83 ở 16 — đọc bảng thuần tuý thì phải chọn 16. Chạy thử trên raw cho kết quả ngược hẳn:

| `phash_dist` | ảnh bị loại ở raw | trong đó `LEAK_TEST` |
|---:|---:|---:|
| 12 | 17 | 5 |
| 16 | 26 | 11 |
| 20 | **134 / 224** | **105** |

Tôi dựng ảnh so sánh từng cặp: 8 trong 9 ảnh bị loại thêm ở mức 16 là **sai hoàn toàn** — `img_0011` (nữ vận động viên
tennis trên sân xanh) bị gán trùng với `img_0008` (hai người trượt tuyết trong tuyết trắng); 6 cờ `LEAK_TEST` mới đều là
ảnh khác hẳn ảnh test (bàn làm việc ↔ trận bóng chày, người đi bộ ↔ xe buýt). Ở mức 20 thì pipeline **xoá 60% lô**.
Calib không thể thấy chuyện này vì hai lý do có thật trong dữ liệu: calib **không có một ca `LEAK_TEST` nào** (nên cột
precision không hề đo chế độ hỏng nguy hiểm nhất là *xoá nhầm ảnh tốt*), và 39 ảnh thì quá ít để hai ảnh bất kỳ tình cờ
có phash gần nhau. **Precision 1,00 trên calib ở đây nghĩa là "chưa đo", không phải "không sai".**

Chỗ thứ hai cần dè chừng: `POSSIBLE_LOOSE_BOX` chỉ có **5 lỗi chèn**, nên mỗi lần bắt trúng/trượt làm recall nhảy 20%.
Bảng sweep ghi recall 0,60 ở conf 0,1 và 0,20 ở conf 0,6 — nghe như mất 2/3, thực chất là **3 box so với 1 box**. Tôi cố ý
không lấy cột này làm căn cứ chọn ngưỡng, mà dựa vào `MISSING` (8 lỗi chèn) và `WRONG_CLASS` (6 lỗi chèn).

## 4. Cửa chặn và điều pipeline KHÔNG bắt được

**(a) Cổng chặn thật.** Chạy không có người duyệt:

```bash
python -m pipeline.run --budget 0 --work work/w_nb --reports work/rep_nb --export-dir work/exp_nb; echo $?
```

→ `kết luận: REWORK — còn 6 lỗi bắt buộc trên 6 mục (MISSING_LABEL_FILE): trả về bước gán nhãn, KHÔNG xuất dữ liệu`,
**mã thoát 20**, và thư mục `work/exp_nb/` **không hề được tạo ra** — không có dataset nào được xuất. Đúng ý đồ: 6 ảnh
thiếu nhãn chỉ người mới gán được, không có người thì lô không đi tiếp, bất kể 196 ảnh còn lại sạch đến đâu.

**(b) Bốn điểm mù, kèm bằng chứng.**

1. **`drop_line` biến ảnh có vật thành "ảnh nền".** Ba ảnh bị autofix xoá hết dòng nhãn: `img_0050` (`1 0.640 0.413 0.000 0.396`
   — w bị làm hỏng thành 0), `img_0104` (id lớp 7), `img_0194` (id lớp 4). Cả ba **vẫn có vật thật**: model lớn thấy *person*
   conf 0,93 đúng ngay tâm box cũ của `img_0050` và `img_0104`. Máy xoá dòng là đúng về cú pháp nhưng **mất thông tin dùng được**
   (tâm và chiều cao của `img_0050` vẫn chuẩn, chỉ w hỏng), rồi file rỗng lại mang nghĩa "ảnh nền". Lần này người cứu được vì
   `FIXED_DROP_LINE` + `EMPTY_LABEL` đẩy chúng lên hạng 16, 17 và 23 — **lọt vào ngân sách 80 trong gang tấc**. Ngân sách nhỏ hơn
   một chút là ba ảnh có người/xe đi thẳng vào dataset với nhãn rỗng. Pipeline không có mã lỗi nào cho tình huống này.
2. **Nhầm lớp mà model lớn cũng nhầm thì không ai bắt.** Trên calib — nơi *đã biết đáp án* — `POSSIBLE_WRONG_CLASS` chỉ đạt
   recall 0,83: 1 trong 6 lỗi nhầm lớp chèn sẵn vẫn lọt. Tệ hơn, lỗi nhầm lớp của calib chủ yếu là person→car (dễ), còn lô raw
   nghiêng về car↔truck↔bus (`img_0020` bus gán truck, `img_0019` car gán truck) — đúng nhóm mà chính yolo26l cũng hay lẫn.
   Recall 0,83 trên calib **không** chuyển thành recall 0,83 trên raw.
3. **Ảnh trùng bị biến đổi mạnh vẫn lọt.** Ở ngưỡng đã chọn, calib còn **2/6 ca `DUP_IMAGE_NEAR` không bắt được** (recall 0,67)
   — là các ca cắt/thu nhỏ mạnh. Không nâng ngưỡng được vì mục 3 đã chứng minh nâng lên 16 là xoá nhầm ảnh tốt. Đây là điểm mù
   phải chấp nhận và ghi nhận, không phải vá bằng ngưỡng.
4. **`BOX_TINY` / `BOX_HUGE` chưa từng được đo.** Hai mã này là quy tắc dự án (`min_box_px: 12`, `max_box_frac: 0.95`), calib
   **không có đáp án** cho chúng. Chúng cờ 6 và 3 dòng trong lô — tôi không có cách nào biết đó là đủ, thừa hay thiếu; `img_0018`
   và `img_0128` (box person chiếm 97–99% ảnh) đúng là lỗi phình box, nhưng không ai đo được số ca bị bỏ sót.

## 5. Train: raw vs clean

**Cách làm cho công bằng.** Cùng `yolo26n.pt` khởi tạo, 20 epoch, batch 16, imgsz 640, cùng `val = gold_val` (40 ảnh nhãn người)
để chọn checkpoint, cùng `test = data/test` (100 ảnh, 387 box) để chấm, **cùng bộ seed 0–4 cho mọi biến thể** (5 seed, không phải 2:
với 2 seed đầu chênh lệch trông như +0,045, thêm seed thì rút xuống +0,028 — tự nó đã là bài học). Chạy local trên RTX 4060,
`scripts/train_compare.py`, mỗi lần ~35 giây. `submission/results/comparison.md`:

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 | precision | recall |
|---|---|---|---|---|---|---|
| raw | 224 | 5 | 0,705 ± 0,022 | 0,504 ± 0,012 | 0,778 ± 0,042 | 0,613 ± 0,035 |
| auto | 196 | 5 | 0,699 ± 0,015 | 0,481 ± 0,015 | 0,742 ± 0,054 | 0,617 ± 0,029 |
| clean | 202 | 5 | **0,733 ± 0,014** | 0,511 ± 0,007 | 0,770 ± 0,050 | 0,648 ± 0,039 |
| oracle | 202 | 5 | 0,698 ± 0,012 | 0,485 ± 0,012 | 0,737 ± 0,036 | 0,622 ± 0,030 |

**Chênh lệch so với nhiễu giữa seed.** `clean − raw = +0,028` mAP50. Độ lệch chuẩn giữa seed là 0,022 (raw) và 0,014 (clean),
nên chênh lệch chỉ lớn hơn nhiễu **khoảng một lần độ lệch chuẩn**: Welch t = 2,38 (df ≈ 6,8), tức **ngay ranh giới p ≈ 0,05**.
Quan trọng hơn con số p: **hai dải giá trị vẫn chồng lấn** — seed tốt nhất của raw (0,734) cao hơn hai seed kém nhất của clean
(0,715 và 0,723). Nếu chỉ chạy một seed và bốc trúng raw-seed2 vs clean-seed3, tôi đã kết luận ngược hoàn toàn.
**Kết luận trung thực: có dấu hiệu nghiêng về `clean`, nhưng với 100 ảnh test và 5 seed thì chưa đủ chắc để tuyên bố
"dữ liệu sau pipeline tốt hơn".**

**Hai nguyên nhân có thể, và tôi đã kiểm bằng thí nghiệm chứ không đoán.**

1. *"Mức tăng là do pipeline lọc dữ liệu hay do 80 ảnh người duyệt?"* — chạy thêm biến thể **`auto`** (y hệt `clean` nhưng
   **không có người duyệt**: `--budget 0 --drop-unlabeled`, 196 ảnh). Kết quả: `auto − raw = −0,006` (t = −0,51) — **phần tự động
   một mình không mang lại lợi ích đo được**, dù nó đã loại 22 ảnh hỏng/trùng/rò rỉ và sửa 35 dòng nhãn. Trong khi đó
   `clean − auto = +0,034` (t = 3,63). So hai manifest: `clean_v1` hơn `clean_auto` đúng **6 ảnh được gán nhãn mới + 74 ảnh được
   sửa nhãn = 80 ảnh người duyệt**. Vậy toàn bộ mức cải thiện đo được nằm ở chỗ **người** chạm vào, đúng luận điểm của bài:
   máy kiểm quy tắc, người kiểm ý nghĩa.
2. *"Có phải raw bị phạt oan vì chứa ảnh rò rỉ test?"* — ngược lại mới đúng. `raw` được train trên **5 ảnh trùng với tập test**
   (`img_0049`↔`test_033`, `img_0101`↔`test_039`, `img_0167`↔`test_044`, `img_0174`↔`test_085`, `img_0221`↔`test_009`), tức điểm
   test của `raw` đang được **thổi lên** một cách không chính đáng — mà vẫn thấp hơn `clean`. Nghĩa là khoảng cách thật có thể
   còn rộng hơn +0,028. Cũng loại được giả thuyết "clean thắng vì ít ảnh hơn nên dễ": `auto` ít ảnh nhất (196) mà không tốt hơn raw.

**Một kết quả tôi chưa giải thích được, ghi lại nguyên trạng.** Biến thể **`oracle`** (đúng 202 ảnh của `clean` nhưng **toàn bộ**
nhãn lấy từ nhãn người) chỉ đạt **0,698 ± 0,012** — **thấp hơn `clean` 0,035** (t = −4,22) và ngang `raw`. Nhãn người lẽ ra phải là
chặn trên. Tôi đã loại một cách giải thích dễ dãi bằng cách đếm: `oracle` có 769 box, 3,81 box/ảnh, gần phân bố tập test nhất
(3,87 box/ảnh) và **ít box nhỏ hơn** `clean` (24 so với 38) — nên không phải "nhãn người nhiều vật khó hơn". Giả thuyết còn lại
của tôi: 122 ảnh không qua người duyệt trong `clean` mang **nhãn do model đã pretrain trên toàn bộ COCO sinh ra**, nên chúng hoạt
động như mục tiêu **chưng cất tri thức** — dễ khớp hơn nhãn người thưa khi chỉ có 202 ảnh và 20 epoch. Nếu đúng thì một phần của
`clean − raw` **không phải là "chất lượng dữ liệu tốt hơn"** mà là hiệu ứng chưng cất, và lợi thế đó sẽ co lại khi train dài hơn.
Cách kiểm tiếp theo: (a) lặp lại với 40 epoch và tập train lớn hơn để xem khoảng cách `clean`−`oracle` có co lại không;
(b) hoán đổi ngẫu nhiên xem 80 ảnh nào được gán nhãn người — nếu hiệu ứng là do chưng cất thì thứ tự hàng đợi không quan trọng;
(c) đo trên tập test lớn hơn 100 ảnh.

**Việc tiếp theo.** Trước khi tin bất kỳ con số nào ở trên để ra quyết định phát hành: tăng số seed lên 10, mở rộng tập test,
và chạy lại `clean` vs `oracle` — vì nếu giả thuyết chưng cất đúng thì chính bộ `clean` đang được chấm điểm cao một cách giả tạo.

## 6. Quyết định và ghi chú AI

**(a) Quyết định: Hold.** Không ký Release, vì `08_summary.json` ghi 18,8% ảnh còn cảnh báo chưa ai xem — gần gấp đôi ngưỡng 10%.
Điều kiện để chuyển sang Release: **tăng ngân sách duyệt lên khoảng 100 ảnh** (hiện 38 ảnh cảnh báo chưa duyệt, cần kéo xuống ≤ 20
ảnh = 10% của 202, tức xem thêm chừng 18–20 ảnh nữa trong 45 ảnh đang bị bỏ lại). Tôi **không** chọn cách rẻ hơn là nâng
`second_opinion.conf` lên 0,8: thí nghiệm cho thấy ở conf 0,8 hàng đợi co còn 102 ảnh và cổng trả về *Release* — nhưng đó là
Release giả, vì recall `POSSIBLE_MISSING` trên calib tụt từ 1,00 xuống 0,75, tức lô "sạch" chỉ vì checker bị bịt mắt.
Nếu phải chốt ngay hôm nay thì phương án trung thực là giữ Hold và giao 45 ảnh còn lại cho vòng duyệt sau, ưu tiên 16 ảnh
`POSSIBLE_LOOSE_BOX`.

**(b) Ghi chú dùng AI.** Có dùng — Claude Code (Opus) để viết 4 module.
- *Prompt chính:* dán nguyên docstring của từng stub + nguyên file test tương ứng, yêu cầu "cài đặt đúng docstring, mọi ngưỡng
  đọc từ `cfg`, không viết cứng số, không xoá/ghi đè gì ngoài `ds`, giải thích từng nhánh bằng tiếng Việt".
- *Cách kiểm:* (1) đọc lại từng dòng, soi riêng chỗ ghi file của `autofix` xem có đụng `data/raw` không; (2) `pytest` 79 test;
  (3) tự dựng thêm 4 ca ngoài test — một dòng dính 2 lỗi, dòng trống + khoảng trắng giữa file, một ảnh có 12 dự đoán, file nhãn
  chỉ gồm dòng trống; (4) đối chiếu với đáp án `data/calib/MANIFEST.csv` qua `make calibrate`.
- *Một lần AI sai:* khi phân tích 3 ảnh bị autofix xoá sạch nhãn, AI kết luận ngay là "chúng đã lọt vào bản xuất dưới dạng ảnh
  nền". Mở `submission/export/clean_v1/labels/` kiểm thì thấy cả 3 đã được người duyệt gán lại nhãn — kết luận ban đầu sai,
  sự thật là chúng **suýt** lọt (hạng 16/17/23 trên ngân sách 80). Bài học: số liệu trong báo cáo phải lấy từ file do pipeline
  sinh ra, không lấy từ suy luận nghe hợp lý.
