# Chạy batch Thực nghiệm 1a (Answer-gain adaptive) trên Gadi

Mục tiêu: so sánh Acc(adaptive) với Acc(Base_fix) cho ba arm Only Pos, Only Neg, Both,
quét ngưỡng max a ∈ {0.25, 0.5, 0.75, 0.9, 1}. Student 8 frame, teacher 32 frame.

## 0. Thông tin cố định

| Mục | Giá trị |
|---|---|
| Project tính SU | `hn98` |
| Queue | `gpuhopper` (1 GPU H200) |
| Code | `/scratch/hn98/$USER/Adaptive-SEED-` (nhánh `exp1-answer-gain-sweep`) |
| venv | `/scratch/hn98/$USER/venv` |
| HF cache (Qwen, BiRefNet) | `/scratch/hn98/$USER/hf_cache` |
| Work-dir | `/scratch/hn98/$USER/runs/exp1` |
| Dữ liệu gốc (chỉ đọc) | `/scratch/jp09/dd9648/datasets_video_hallu` |

Node GPU **không có Internet**. Mọi thứ phải tải trước ở login node.

## 1. Kiểm tra trước khi submit (login node)

Lấy code mới nhất:

```bash
cd /scratch/hn98/$USER/Adaptive-SEED-
git pull mine exp1-answer-gain-sweep
git log --oneline -1
```

Kiểm tra dữ liệu trong work-dir:

```bash
W=/scratch/hn98/$USER/runs/exp1
find $W/VideoHallucer -name "*.mp4" | wc -l                       # mong đợi 348
for d in entire interleave misleading; do echo $d $(ls $W/EventHallusion/$d | wc -l); done   # 109 / 193 / 95
ls $HF_HOME/hub 2>/dev/null || ls /scratch/hn98/$USER/hf_cache/hub   # có models--Qwen... và models--ZhengPeng7--BiRefNet
```

Kiểm tra SU còn lại và không có job cũ đang chạy:

```bash
nci_account -P hn98
qstat -u $USER
```

Chỉ chạy **một tiến trình tại một thời điểm** trên cùng work-dir, vì các job dùng chung cache
(mask foreground, logit teacher). Không submit song song.

## 2. Submit

### 2.1 Chạy thử một ngưỡng (khuyến nghị làm trước)

```bash
cd /scratch/hn98/$USER/Adaptive-SEED-
qsub -v ARM=adaptive_pos_only,VALUES=0.9 scripts/run_exp1.pbs
```

Job này dựng toàn bộ cache dùng chung (mask foreground 32 frame, logit teacher, điểm base)
và kiểm tra pipeline end-to-end. Kết quả ngưỡng 0.9 được dùng lại ở lượt chạy đầy đủ.

### 2.2 Chạy đầy đủ, ba job nối tiếp

```bash
J1=$(qsub -v ARM=adaptive_pos_only scripts/run_exp1.pbs)
J2=$(qsub -W depend=afterok:$J1 -v ARM=adaptive_neg_only scripts/run_exp1.pbs)
J3=$(qsub -W depend=afterok:$J2 -v ARM=adaptive_pos_neg scripts/run_exp1.pbs)
echo $J1 $J2 $J3
```

Ý nghĩa từng arm (mỗi lần chạy gồm `seed_base` = Base_fix cùng a_p, a_n, và arm đang thử):

| `ARM` | Quét | Giá trị cố định |
|---|---|---|
| `adaptive_pos_only` | a_p = giá trị quét | a_n = 0.25 |
| `adaptive_neg_only` | a_n = giá trị quét | a_p = 0.9 |
| `adaptive_pos_neg` | a_p = a_n = giá trị quét | không |

### 2.3 Biến môi trường tuỳ chọn (truyền qua `qsub -v`, phân tách bằng dấu phẩy)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `VALUES` | `0.25 0.5 0.75 0.9 1` | Chỉ chạy một phần ngưỡng, ví dụ `VALUES=0.5` |
| `FIXED_AP` | `0.9` | a_p khi chỉ quét a_n |
| `FIXED_AN` | `0.25` | a_n khi chỉ quét a_p |
| `TEACHER_FRAMES` | `32` | Số frame của teacher |
| `WORK` | `/scratch/hn98/$USER/runs/exp1` | Work-dir |

Ví dụ: `qsub -v ARM=adaptive_pos_only,VALUES="0.25 0.5" scripts/run_exp1.pbs`

## 3. Theo dõi

```bash
qstat -u $USER                      # Q = đang chờ, R = đang chạy, H = bị giữ (chờ job trước)
tail -n 40 seed_exp1.o<jobid>       # log nằm ở thư mục đã submit
qstat -fx <jobid> | grep -iE "estimated|comment|resources_used"
```

Các mốc cần thấy trong log, theo thứ tự:

1. `student n_frames = 8 | teacher_frames = 32`
2. `HF_HUB_OFFLINE=1: dùng dữ liệu local ...` và `train 150 cặp | val 50 cặp | test 155 cặp`
3. `EventHallusion: ... câu / ... video | thiếu: 0`
4. `fg cache: ... | lưới (16, 16, 16)` (BiRefNet; lần đầu mất khoảng 30 phút cho cả VH và EH)
5. `null positive vs orig: max|Δ| = 0.00e+00`
6. Thanh `Shared teacher views TRAIN` (tính teacher 32 frame, chỉ một lần)
7. `[seed_base_s0] epoch 0: {...}` rồi `[adaptive_pos_only_s0] epoch 0: {...}`
8. Với mỗi ngưỡng, dòng `===== <ARM> | a_pos=... a_neg=... =====` mở đầu một lượt

Nếu một ngưỡng lỗi, script chỉ in `FAILED: a_pos=... a_neg=...` rồi chạy tiếp ngưỡng sau.
Tìm nhanh các ngưỡng lỗi:

```bash
grep -n "FAILED" seed_exp1.o*
```

## 4. Kết quả

Mỗi cấu hình có một thư mục riêng:

```
/scratch/hn98/$USER/runs/exp1/cdt_distill_cache_f8_448/seed_component_ablation/runs/ap<a>_an<b>_tf32/
├── results.csv                    # q_acc, pair_acc, yes_rate, delta_pp, wrong_to_correct, correct_to_wrong
├── target_diagnostics_train.csv   # hệ số a_p, a_n thực tế theo từng mẫu train
└── <arm>_s0_predictions.csv       # margin theo từng câu hỏi
```

Các split trong `results.csv`: `VH_VAL`, `VH_TEST` (cột `pair_acc` có giá trị), `EH_ALL`,
`eh_entire`, `eh_misleading`, `eh_mix`.

Gộp tất cả cấu hình thành một bảng để điền vào bảng báo cáo:

```bash
source /scratch/hn98/$USER/venv/bin/activate
python - <<'EOF'
import glob, os, pandas as pd
root = f"/scratch/hn98/{os.environ['USER']}/runs/exp1/cdt_distill_cache_f8_448/seed_component_ablation/runs"
frames = []
for f in sorted(glob.glob(f"{root}/*/results.csv")):
    d = pd.read_csv(f)
    d['config'] = os.path.basename(os.path.dirname(f))
    frames.append(d)
df = pd.concat(frames)
keep = df[df.split.isin(['VH_TEST', 'EH_ALL'])]
tab = keep.pivot_table(index=['config', 'run'], columns='split', values=['q_acc', 'pair_acc'], aggfunc='first')
print(tab.round(4).to_string())
df.to_csv(f"/scratch/hn98/{os.environ['USER']}/runs/exp1/all_results.csv", index=False)
EOF
```

So sánh Acc(adaptive) > Acc(Base_fix): với mỗi `config`, so hàng `adaptive_*_s0` với hàng `seed_base_s0`.

## 5. Chạy lại và tiếp tục khi bị gián đoạn

- Mask foreground, logit teacher, điểm `base` và student đã train xong đều được cache.
  Submit lại cùng lệnh là bỏ qua phần đã xong và chỉ làm phần còn thiếu.
- Một lần train bị ngắt giữa epoch **không** có resume; nó train lại từ đầu lần đó.
- Sửa bất kỳ file `.py` nào trong `src/` làm hash mã nguồn đổi, các cache gắn với hash (logit teacher, student) sẽ tính lại.
  Mask foreground thì không bị ảnh hưởng.

## 6. Xử lý lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `qsub: Job rejected ... queue` | Project không có quyền queue. Thử `-P jp09` trong file `.pbs`, giữ `storage=scratch/hn98+scratch/jp09` |
| Job `Q` rất lâu | Hàng đợi H200 kín (120 GPU). Xem `qstat -Q gpuhopper`. Có thể đổi sang `dgxa100` (`ncpus=16`, `mem=120GB`) |
| `/scratch/hn98: No such file` trong job | Thiếu `scratch/hn98` trong dòng `#PBS -l storage=` |
| `Network is unreachable` / lỗi tải model | Node GPU không có Internet. Cần `HF_HUB_OFFLINE=1` và model đã nằm trong `$HF_HOME` |
| `EventHallusion thiếu N video` | Thiếu video trong `$W/EventHallusion/`. Dựng lại bằng `cp -rs` từ bản đầy đủ (xem mục 1) |
| `Disk quota exceeded` | Home đầy (10 GB). Kiểm tra `quota -s`; đặt cache vào `/scratch` |
| Hết walltime | Tăng `#PBS -l walltime`, submit lại, phần đã cache được dùng lại |
| Job sau bị `H` mãi | Job trước thất bại nên `afterok` không thoả. `qdel` job bị giữ và submit lại |

Dừng một job:

```bash
qdel <jobid>
```

## 7. Dọn dẹp

Sau khi lấy xong kết quả, kiểm tra dung lượng của project:

```bash
lquota
du -sh /scratch/hn98/$USER/runs/exp1/cdt_distill_cache_f8_448/*
```

Không xoá thư mục `fg_*` và `seed_component_ablation/views_*` nếu còn định chạy thêm ngưỡng,
vì tính lại tốn nhiều giờ GPU.