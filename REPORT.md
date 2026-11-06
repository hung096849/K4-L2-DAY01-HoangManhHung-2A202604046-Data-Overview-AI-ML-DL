# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.x / 2.x / 8.4.145
**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

> File này là bản **đã điền** để dùng thử upload Colab / nộp `report/`.
> Số liệu record dưới đây là ví dụ điển hình cho sample `traffic` / `kitchen`; nếu JSON máy bạn
> ra khác thì thay đúng số từ file output của bạn. Không ghi họ tên, MSSV hay dữ liệu cá nhân.

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

1. Record hạng 1 của ảnh `traffic`:
   - `class_id` = 468
   - `class_name` = cab
   - `rank` = 1
   - `score` = 0.510915 (khoảng 51%)
   - `taxonomy_name` = ImageNet-1K

2. Record này nói về **cả bức ảnh**. Trong record không có toạ độ hay hộp nào, chỉ có một tên lớp (kèm score) gắn cho toàn bộ ảnh. Vì vậy model không chỉ ra vật nào nằm ở đâu, mà chỉ trả lời "bức ảnh này thuộc lớp nào".

3. Danh sách lớp **đi kèm taxonomy/checkpoint**, model không tự nghĩ ra. Checkpoint `yolo11n-cls.pt` được huấn luyện sẵn với danh sách cố định 1000 lớp của ImageNet-1K, nên nó chỉ có thể chọn trong danh sách đó. Đây cũng là lý do top-1 của ảnh đường phố COCO có thể trông "lạ": ảnh chứa thứ ngoài danh sách thì model vẫn buộc phải chọn lớp gần nhất. Ở ảnh `traffic`, top-5 gồm `cab` (0.51), `minibus` (0.16), `police_van` (0.09), `recreational_vehicle` (0.05), `streetcar` (0.05): tất cả đều là tên có sẵn trong ImageNet-1K.

4. Cần ghi đủ ba trường vì mỗi trường có một vai trò:
   - `class_id` là số máy dùng để lưu và tính toán.
   - `class_name` là tên để người đọc hiểu.
   - `taxonomy_name` cho biết số và tên đó thuộc bảng lớp nào. Cùng một `class_id` có thể mang nghĩa khác nhau ở taxonomy khác (ví dụ id 0 là `tench` trong ImageNet-1K nhưng là `person` trong COCO-80).

   Nếu chỉ ghi tên lớp thì dễ nhầm lẫn khi trộn dữ liệu hoặc đổi model.

5. Khi ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần nói rõ **tiêu chí chọn đúng một lớp**, ví dụ: chọn chủ thể chính (vật chiếm diện tích lớn nhất hoặc nổi bật nhất); khi hai chủ thể ngang nhau thì ưu tiên theo một thứ tự đã quy định; có lớp dự phòng như `other/unclear` cho ảnh không rõ; và ca nào guideline chưa nói tới thì escalation thay vì tự đoán. Nếu không có quy tắc này, mỗi người gán sẽ chọn một kiểu và nhãn sẽ không nhất quán.

6. Không được lấy `score` làm ground truth vì score chỉ là mức "tự tin" của model, mà model có thể **sai nhưng vẫn rất tự tin** (như ví dụ ảnh COCO bị gán lớp ImageNet). Ground truth phải do người gán theo guideline. Ví dụ ở ảnh `traffic`, model chỉ tự tin khoảng 51% với `cab`, còn cùng ảnh đó model detection lại thấy `bus`, `car`, `person`; điểm số cho thấy model chưa chắc chắn chứ không chứng minh `cab` là nhãn đúng. Nếu dùng dự đoán của model làm nhãn chuẩn để huấn luyện lại, model chỉ học lại chính lỗi của nó và không có ai kiểm chứng.

## 2. Phát hiện vật thể

1. Một record của ảnh `kitchen`:
   - `class_name` = person
   - `score` = 0.912625
   - `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92]
   - `bbox_width` = 113.58
   - `bbox_height` = 279.68

2. Box có dạng `[x_min, y_min, x_max, y_max]`, đơn vị **pixel**, gốc `(0, 0)` ở **góc trên-trái** của ảnh (ảnh `kitchen` rộng 640, cao 427 pixel). Với record trên:
   - Góc trên-trái nằm tại `(x_min, y_min)` = (385.33, 69.24).
   - Góc dưới-phải nằm tại `(x_max, y_max)` = (498.92, 348.92).
   - Chiều rộng = 498.92 − 385.33 ≈ 113.58 pixel, chiều cao = 348.92 − 69.24 = 279.68 pixel (khớp `bbox_width`, `bbox_height`; chênh 0.01 ở chiều rộng là do làm tròn).

   Lưu ý trục y tăng **xuống dưới**, nên "trên-trái" có toạ độ nhỏ nhất. Hộp này cao gần gấp 2,5 lần bề rộng, phù hợp với dáng một người đứng.

3. Số prediction theo ngưỡng, ảnh `kitchen`:
   - Ngưỡng 0.35: 11 vật thể (5 bowl, 2 oven, 2 person, 2 cup).
   - Ngưỡng 0.60: 6 vật thể (2 person, 2 bowl, 2 oven; tức chỉ còn các hộp có score từ 0.61 trở lên).

   Khi tăng ngưỡng từ 0.35 lên 0.60, 5 hộp biến mất: 3 bowl (score 0.50, 0.46, 0.38) và 2 cup (score 0.45, 0.38). Đó đều là vật nhỏ, và nếu chúng là vật thật thì ngưỡng cao làm sót. Ngược lại, ngưỡng càng thấp thì càng nhiều hộp, kèm nguy cơ có hộp sai hoặc trùng.

4. Khi hạ ngưỡng và giữ nhiều hộp hơn, **reviewer phải làm thêm việc**: kiểm tra từng hộp thừa xem có phải vật thật không, loại hộp sai hoặc trùng lặp, sửa tên lớp nhận nhầm và chỉnh lại hộp bị lệch. Khối lượng kiểm tra và rework tăng theo số hộp.

5. **Quy tắc box:** Vẽ một hộp chữ nhật cho mỗi vật thể thuộc taxonomy, bám sát mép ngoài của phần **nhìn thấy được** của vật (bao trọn vật, không cắt mất phần nào, không chừa quá nhiều nền thừa), và không vẽ một hộp chung cho nhiều vật.

   Quy tắc này không phụ thuộc điểm tin cậy: người gán phải gán đủ mọi vật thuộc taxonomy, kể cả vật nhỏ hoặc khó thấy, và không bỏ vật chỉ vì "model sẽ không đoán ra".

6. Vật bị cắt mép trong ảnh `kitchen`: record `person` thứ hai, `bbox_xyxy` = [0.12, 263.18, 61.55, 310.78] (score 0.61). Hộp bắt đầu từ `x_min` = 0.12, tức sát mép trái ảnh, và chỉ cao 47.59 pixel (so với người còn lại cao 279.68 pixel), nên đây là một người chỉ lộ một phần ở mép ảnh. Một số hộp khác cũng chạm mép trái (`oven` có `x_min` = 0.11, `bowl` có `x_min` = 0.32).

   Cách xử lý: trường hợp **bị cắt mép ảnh hoặc bị che một phần** có thể viết thành quy tắc trong guideline, ví dụ "vẽ hộp quanh phần nhìn thấy được, kể cả khi vật chỉ lộ một phần", vì đây là tình huống lặp lại nhiều lần ở nhiều ảnh. Còn nếu vật bị che hoặc cắt đến mức **không chắc là vật gì** (ví dụ chỉ thấy một mảnh nhỏ) hoặc guideline chưa có quy định thì phải **escalation** hỏi Lab Coach, không tự bịa nhãn, vì mỗi người đoán một kiểu sẽ làm nhãn không nhất quán.

## 3. Phân đoạn theo từng đối tượng

1. Một record của ảnh `kitchen`:
   - `instance_id` = kitchen-001
   - `class_name` = person
   - `score` = 0.899318
   - `polygon_point_count` = 348
   - 4 điểm đầu của `polygon_xy` = [446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0]

2. So với box, polygon cho thêm thông tin về **hình dạng thật của vật thể**: đường viền bám theo biên, từ đó biết vật chiếm đúng vùng pixel nào. Ví dụ người `kitchen-001` có hộp `bbox_xyxy` = [385.45, 66.44, 498.02, 348.58] chỉ là một hình chữ nhật, còn polygon gồm 348 điểm bám theo đường viền cơ thể (đầu, vai, tay, chân), nên nó cho biết dáng người chứ không chỉ vị trí. Box chỉ là hình chữ nhật bao quanh nên chứa cả nền thừa ở các góc, còn polygon loại phần nền này ra, nhờ đó tách được các vật chồng sát nhau và tính được diện tích vật chính xác hơn.

3. `instance_id` dùng để **phân biệt từng vật cụ thể** trong output của bài lab, nên hai vật cùng lớp (ví dụ hai cái ly) vẫn có hai `instance_id` và hai polygon riêng. Câu "không phải class ID và không phải tracking ID" nghĩa là:
   - **Không phải class ID**: class ID cho biết vật thuộc *loại* nào (nhiều vật có thể trùng class ID), còn `instance_id` là mã riêng của *từng vật*.
   - **Không phải tracking ID**: tracking ID dùng để theo dõi cùng một vật qua nhiều khung hình video, còn `instance_id` chỉ có ý nghĩa trong một ảnh của bài lab này, không liên quan đến việc theo dõi.

4. **Quy tắc vẽ biên mask:** Vẽ polygon bám sát đường biên ngoài của phần nhìn thấy được của vật, đặt đủ điểm ở chỗ biên cong hoặc có góc để hình không bị "gãy" hay cắt mất phần vật, không để lọt nền vào trong, và mỗi vật là một polygon riêng (kể cả khi hai vật cùng lớp đứng sát nhau).

5. Chỗ khó trong ảnh `kitchen` nằm ở **góc dưới bên trái**, nơi nhiều vật chồng lên nhau: ba `bowl` (kitchen-002 score 0.74, kitchen-003 score 0.70, kitchen-006 score 0.53), `person` 0.43 (kitchen-009), `oven` 0.49 (kitchen-007) và `dining table` 0.58 (kitchen-005). Các vùng mask và hộp của chúng giao nhau, nhãn chữ che lên nhau, và `oven` với `dining table` chồng lấn lên cùng một vùng ảnh nên khó biết ranh giới thật nằm ở đâu. Ngoài ra `potted plant` 0.63 ở góc trên trái thực chất trông như một bó cây khô treo, không phải cây trồng trong chậu, nên chọn lớp nào cũng là ca khó.

   Cách xử lý: các vật **chạm nhau hoặc che nhau** là tình huống thường gặp, nên viết thành **guideline** để mọi người làm giống nhau (mỗi vật một polygon riêng, vẽ ranh giới ở chỗ tiếp xúc, vật bị che thì chỉ vẽ phần nhìn thấy, không để hai polygon chồng lên cùng một vùng). Còn những ca **không xác định được lớp hoặc ranh giới** như bó cây khô bị gọi là `potted plant`, hoặc vùng `oven`/`dining table` chồng lấn không rõ, thì phải **escalation** cho Lab Coach, vì tự quyết theo cảm tính sẽ cho nhãn không nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp (`class_id`, `class_name`, thuộc `taxonomy_name`) cho **cả ảnh** | Ảnh `traffic` có nhiều vật (theo `detection_predictions.json`: bus, car, person) nhưng classification chỉ cho một lớp top-1 `cab` với score 0.51; các hạng sau (`minibus` 0.16, `police_van` 0.09...) cũng đều là xe, nên một nhãn không mô tả hết cảnh và lớp "đúng" phụ thuộc quy tắc chọn chủ thể của guideline | Chọn đúng một lớp cho ảnh theo quy tắc chủ thể chính của guideline | Kiểm tra lớp có đúng taxonomy và đúng quy tắc chọn chủ thể không; không dùng score của model làm căn cứ đúng/sai |
| Phát hiện vật thể | Một **hộp** `[x_min, y_min, x_max, y_max]` (pixel) kèm lớp cho **mỗi vật thể** | Ảnh `traffic`: có record `person` với `bbox_xyxy` bắt đầu từ x_min = 0.08 và rộng chỉ 18.32 pixel, tức người bị cắt ở sát mép trái ảnh và rất nhỏ, khó quyết định có gán không và hộp lấy đến đâu | Vẽ hộp bám sát vật, gán đúng lớp, gán đủ mọi vật thuộc taxonomy | Kiểm tra hộp có bao trọn vật, có sót hoặc thừa hộp, có đúng lớp không |
| Instance segmentation | Một **polygon** `polygon_xy` kèm lớp và `instance_id` riêng cho **mỗi vật thể** | Ảnh `traffic`: polygon của bus `traffic-001` có 120 điểm toạ độ nguyên tạo thành đường răng cưa (ví dụ (148,189) → (147,190) → (145,190)), và `bbox_xyxy` của nó [95.3, 188.72, 224.08, 319.96] lệch nhẹ so với hộp detection của cùng chiếc bus [93.17, 187.95, 223.01, 320.91], nên vị trí biên chỉ là xấp xỉ | Vẽ viền bám sát biên từng vật, tách riêng từng vật cùng lớp | Kiểm tra viền có bám biên, có lọt nền, có gộp hai vật làm một, có đủ điểm ở chỗ cong không |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai mà bài lab cung cấp, không tải ảnh cá nhân hay dữ liệu nội bộ lên, và không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo hoặc output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (ví dụ ảnh cá nhân, ảnh nhạy cảm, dữ liệu nội bộ), mình sẽ dừng lại, không xử lý tiếp và báo cho Lab Coach/giảng viên (mentor) để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
