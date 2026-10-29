# SEED Video Distillation

Python package cho thí nghiệm positive/negative distillation trên video, được tách từ `SEED_Component_Ablation (4).ipynb`. Bao gồm toàn bộ thành phần và add-on của notebook; không cần notebook để chạy.

## Cài đặt

Dùng Python >= 3.10 và GPU CUDA hỗ trợ BF16. Cài PyTorch/torchvision tương thích CUDA của máy trước, sau đó:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
seed-video --help
```

`transformers==4.51.3`, `peft==0.15.2` được giữ từ notebook. Các dependency còn lại chưa được khóa thành môi trường tái lập đầy đủ. Nếu môi trường Colab có torchao cũ gây lỗi PEFT, gỡ phiên bản torchao xung đột trước. Dataset/model được tải khi chạy; cần Internet và đủ dung lượng lưu trữ.

## Chạy

```bash
# Xem thứ tự các stage, không tải dữ liệu hay model
seed-video --stage all --dry-run

# 8 cấu hình ablation: base SEED, token, pair, pos, neg, token+pair, pos+neg, all
seed-video --stage core --work-dir runs/core

# Core + answer-token/agreement + keep/repair không caption
seed-video --stage no-caption --work-dir runs/no_caption

# Core + caption teacher32 + factorial FULL
seed-video --stage full --work-dir runs/full

# Toàn bộ theo thứ tự notebook, gồm cả SeedNoise
seed-video --stage all --work-dir runs/all
```

Có thể dùng `python -m seed_video` thay `seed-video`. Thêm `--no-eh` để bỏ đánh giá EventHallusion. Trên Colab, mount Drive trước nếu chọn `--work-dir` nằm trên Drive.

## Các module

| Đường dẫn trong `src/seed_video/` | Trách nhiệm |
|---|---|
| `cli.py`, `pipeline.py` | Tham số CLI, thứ tự thực thi và fingerprint mã nguồn |
| `config.py` | Cấu hình mô hình, dữ liệu, LoRA, SEED, đường dẫn |
| `runtime.py` | Trạng thái riêng cho mỗi lần chạy |
| `data/videohallucer.py` | Tải dữ liệu, split theo video |
| `data/eventhallusion.py` | Nạp và kiểm tra benchmark ngoài |
| `data/video.py` | Decode frame, tạo view orig/TCD/shuffle/noise |
| `models/foreground.py` | BiRefNet và cache foreground |
| `models/qwen.py` | Qwen, visual hook, forward theo view |
| `components/adaptive.py` | Hệ số adaptive từ answer gain |
| `components/gates.py` | JSD, pair gate, correct-answer margin |
| `components/positive.py` | Feature Enhance và mask ngẫu nhiên |
| `teachers/` | Cấu hình nhánh, tổ hợp logit teacher |
| `training/losses.py` | SEED/KD và các loss hỗ trợ của nguồn |
| `training/trainer.py` | LoRA, vòng train, checkpoint |
| `evaluation/` | Inference, q/pair accuracy, bảng kết quả |
| `experiments/components.py` | Ablation chính và visual-token selection |
| `experiments/seednoise.py` | Thay negative TCD bằng SeedNoise |
| `experiments/answer_tokens.py` | Answer-token selection và agreement |
| `experiments/caption.py` | Caption teacher 32 frame, null/reverse regularization |
| `experiments/keep_repair.py` | Keep/repair không caption |
| `experiments/factorial.py` | Factorial adaptive × keep/repair, FULL |

Các hàm tính toán nhận `runtime` tường minh; các `run(runtime)` khởi tạo từng stage. Các callback được bind vào runtime để các thí nghiệm vẫn thay thế teacher/step/config như notebook, nhưng không dùng namespace toàn cục, `exec`, hoặc `import *`. Import package không tải model, không tải dữ liệu và không chạy train. Đây là cấu trúc stage cho nghiên cứu; runtime vẫn chứa trạng thái động để bảo toàn các add-on gốc.

Đổi hyperparameter chung trong `config.py`; thông số riêng nằm ở đầu hàm `run()` của từng experiment. Mặc định kế thừa FAST: 8 frame, TCD 4 frame, 1 epoch; caption teacher 32 frame ở stage tương ứng.

## Quy trình và giới hạn

- Adaptive answer-gain của ablation chính dùng nhãn TRAIN. Không gọi đây là phương pháp hoàn toàn không nhãn.
- Pair gate mặc định là câu hỏi proxy chuyển động trái/phải; không bảo đảm equivariance đúng cho mọi câu hỏi VideoHallucer.
- Giữ logic split, loss và công thức từ nguồn. Kết quả số cần được xác nhận lại sau refactor.
- Checkpoint cache hoàn thành có thể tái sử dụng; run bị gián đoạn giữa epoch không có optimizer-resume tự động.
- Fingerprint tính trên toàn bộ mã package để tránh dùng checkpoint của bản code khác.
- Các add-on có thể cần nhiều RAM/VRAM và nhiều lượt train; `all` không chỉ chạy một cấu hình FULL.

## Kết quả

CSV, teacher cache và adapter nằm dưới `--work-dir/cdt_distill_cache_f8_448/`, trong thư mục riêng của từng thí nghiệm. Báo cáo gồm accuracy, pair accuracy cho VH, delta, wrong-to-correct và correct-to-wrong. Không đưa dữ liệu/model/checkpoint lên GitHub; `.gitignore` đã loại các loại này.

## Kiểm tra

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m seed_video --stage all --dry-run
python -m compileall -q src
```

Đã kiểm tra cấu trúc, import không side effect, thứ tự phụ thuộc, cách bind callback và biên hệ số adaptive trên CPU không cần torch. Chưa chạy end-to-end với model 7B/GPU, chưa xác nhận parity thực nghiệm với notebook.

## Đưa lên GitHub

Giải nén dự án, tạo repository GitHub trống, rồi chạy trong thư mục dự án:

```bash
git init
git add .
git commit -m "Organize SEED video distillation experiments"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Thay URL bằng repository của bạn. Chưa gán giấy phép mã nguồn; chủ sở hữu cần chọn license phù hợp trước khi phát hành công khai. Không kèm dữ liệu hoặc trọng số của bên thứ ba.
