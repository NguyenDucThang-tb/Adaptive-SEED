# Thực nghiệm 1a: Answer-gain adaptive, chạy trên Gadi

Mục tiêu: so sánh Acc(adaptive) với Acc(Base_fix) cho ba arm Only Pos, Only Neg, Both, quét
ngưỡng max của a_p và a_n ∈ {0.25, 0.5, 0.75, 0.9, 1}. Student 8 frame, teacher 32 frame.

Kết quả được báo cáo theo thứ tự cột của bảng thực nghiệm:

| Arm | VideoHallucer pair acc | VideoHallucer Q acc | EventHallusion |
|---|---|---|---|

Cả ba chỉ số (cùng wrong-to-correct và correct-to-wrong) được tính trong **cùng một lần chạy**
mỗi cấu hình; không có lần chạy riêng cho từng chỉ số. Thứ tự chỉ ảnh hưởng cách gộp bảng ở mục 6.

## 0. Quy ước chung

Chạy từng khối lệnh trên login node của Gadi. **Không dán cả file vào terminal.**

| Mục | Giá trị |
|---|---|
| Project tính SU | `hn98` |
| Queue | `gpuhopper` (1 GPU H200) |
| Code | `/scratch/hn98/$USER/Adaptive-SEED` |
| venv | `/scratch/hn98/$USER/venv` |
| HF cache (Qwen, BiRefNet) | `/scratch/hn98/$USER/hf_cache` |
| Dữ liệu dùng chung | `/scratch/hn98/$USER/data` |
| Work-dir (cache, checkpoint, kết quả) | `/scratch/hn98/$USER/runs/exp1` |

Node GPU **không có Internet**. Mọi thứ cần tải phải tải ở login node trước.
Job chỉ ghi vào `/scratch`, không ghi vào `$HOME` (quota 10 GB).

```bash
export CODE=/scratch/hn98/$USER/Adaptive-SEED
export DATA=/scratch/hn98/$USER/data
export WORK=/scratch/hn98/$USER/runs/exp1
export HF_HOME=/scratch/hn98/$USER/hf_cache
export PIP_CACHE_DIR=/scratch/hn98/$USER/pip_cache
mkdir -p $DATA $WORK $HF_HOME $PIP_CACHE_DIR /scratch/hn98/$USER/logs
```

Các biến này mất khi đóng terminal; chạy lại khối trên mỗi phiên mới.

## 1. Chuẩn bị một lần

### 1.1 Code

```bash
cd /scratch/hn98/$USER
git clone https://github.com/NguyenDucThang-tb/Adaptive-SEED.git
cd $CODE && git checkout exp1-answer-gain-sweep && git log --oneline -1
```

Cập nhật sau này: `git pull`. **Không `git pull` hoặc sửa file `.py` giữa các lần resume** của cùng một
cấu hình, vì hash mã nguồn đổi làm cache teacher và checkpoint cũ không còn được nhận (mục 7).

### 1.2 Môi trường Python

```bash
module load python3/3.11.7 cuda/12.3.2
python3 -m venv /scratch/hn98/$USER/venv
source /scratch/hn98/$USER/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r $CODE/requirements.txt
pip list 2>/dev/null | grep -iE "^(torch|transformers|peft) "
```

Cần `transformers >=4.57.0,<5` và `peft 0.15.2` cho Qwen3-VL. Repo không có `pyproject.toml`; code chạy bằng `PYTHONPATH=src`
(file `.pbs` đã đặt sẵn).

### 1.3 Model (tải ở login node)

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
huggingface-cli download ZhengPeng7/BiRefNet
ls $HF_HOME/hub
```

Phải thấy `models--Qwen--Qwen3-VL-8B-Instruct` và `models--ZhengPeng7--BiRefNet`.
Nên chạy trong `screen -S dl` để không mất khi rớt SSH.

Qwen3-VL dùng vision patch stride 32 và model class riêng. Không tiếp tục dùng venv Transformers 4.51.3; tạo venv mới hoặc nâng cấp môi trường trước khi submit job. `config.py` mặc định đã chọn Qwen3-VL-8B-Instruct. Model change làm thay đổi cache key và fingerprint, vì vậy kết quả/checkpoint Qwen2.5 cũ không được resume vào run Qwen3.

### 1.4 Dữ liệu VideoHallucer

Dùng lại bản có sẵn của project `jp09` (chỉ đọc), liên kết từng file bằng `cp -rs`:

```bash
SRC_VH=/scratch/jp09/dd9648/datasets_video_hallu/videohallucer
mkdir -p $WORK/VideoHallucer
cp -rs $SRC_VH/temporal $SRC_VH/object_relation $WORK/VideoHallucer/
find $WORK/VideoHallucer -name "*.mp4" | wc -l      # mong đợi 348
```

### 1.5 Dữ liệu EventHallusion (cần đủ 109 / 193 / 95 video)

Bản có sẵn ở `jp09` **thiếu 40 video `mix`**. Tải bản đầy đủ từ Google Drive (id ghim trong `config.py`):

```bash
mkdir -p $DATA/eh && cd $DATA/eh
gdown 1IPmx6Y80UrXwVPmZJh6zjCPHtlsw4p9n -O videos.zip
python - <<'EOF'
import zipfile
z = zipfile.ZipFile('videos.zip')
print('entry:', len(z.infolist()), '| giải nén: %.2f GB' % (sum(i.file_size for i in z.infolist()) / 1e9))
EOF
```

Tổng giải nén phải gần 5 GB. Bản `unzip` mới báo nhầm "zip bomb" với file này, nên giải nén bằng Python:

```bash
python - <<'EOF'
import zipfile
zipfile.ZipFile('videos.zip').extractall('extracted')
EOF
for d in entire interleave misleading; do echo $d $(ls extracted/videos/$d | wc -l); done   # 109 / 193 / 95
mkdir -p $DATA/eh/questions
cp /scratch/jp09/dd9648/video-hallucination-benchmark/external/EventHallusion/questions/*.json $DATA/eh/questions/
ls $DATA/eh/questions
```

Phải có `entire_questions.json`, `misleading_questions.json`, `mix_questions.json`.

### 1.6 Dựng work-dir

Hàm sau dựng EventHallusion vào một work-dir bất kỳ (dùng lại khi tạo work-dir riêng cho smoke test hoặc chạy song song):

```bash
prep_eh() {   # prep_eh <work-dir>
  mkdir -p "$1/EventHallusion/questions"
  cp $DATA/eh/questions/*.json "$1/EventHallusion/questions/"
  cp -rs $DATA/eh/extracted/videos/. "$1/EventHallusion/"
  for d in entire interleave misleading; do echo "$d $(ls "$1/EventHallusion/$d" | wc -l)"; done
}
prep_eh $WORK
```

Phải in `entire 109`, `interleave 193`, `misleading 95`. Nếu thiếu `questions/*.json`, code sẽ cố tải từ
GitHub và job chết với `Network is unreachable`.

### 1.7 Kiểm tra SU

```bash
nci_account -P hn98
```

Cột `Avail` phải còn đủ SU. Mỗi giờ H200 tính theo hệ số của queue `gpuhopper`.

## 2. Smoke test (làm trước khi chạy thật)

Chạy 1 cặp train, 1 val, 1 test, chỉ subset `temporal`, tắt EventHallusion. Mục đích là kiểm tra pipeline
(dữ liệu, mask, Qwen, teacher 32 frame, train, đánh giá), không dùng accuracy của smoke làm kết quả.

```bash
cd $CODE
export SMOKE=/scratch/hn98/$USER/runs/exp1_smoke
mkdir -p $SMOKE && ln -s $WORK/VideoHallucer $SMOKE/VideoHallucer
qsub -l walltime=00:30:00 -v ARM=adaptive_pos_only,VALUES=0.9,SMOKE_PAIRS=1,WORK=$SMOKE scripts/run_exp1.pbs
```

Nếu `ln -s` báo đích đã tồn tại, kiểm tra thư mục trước, đừng ghi đè. Theo dõi:

```bash
qstat -u $USER
tail -n 40 seed_exp1.o*
```

Smoke đạt khi thấy theo thứ tự: `SMOKE mode: temporal only ...`, `fg cache: ... lưới (16, 16, 16)`,
`null positive vs orig: max|Δ| = 0.00e+00`, `Shared teacher views TRAIN`, `[seed_base_s0] epoch 0`,
`[adaptive_pos_only_s0] epoch 0`, và cuối log có `Exit Status: 0`.

## 3. Chạy đầy đủ

Mỗi arm là một chuỗi job 3 giờ nối nhau. Cả ba arm dùng chung `WORK` nên **chạy nối tiếp**, không song song.

| `ARM` | Bảng | Quét | Cố định |
|---|---|---|---|
| `adaptive_pos_only` | Only Pos | a_p = giá trị quét | a_n = 0.25 |
| `adaptive_neg_only` | Only Neg | a_n = giá trị quét | a_p = 0.9 |
| `adaptive_pos_neg` | Both | a_p = a_n = giá trị quét | không |

Mỗi lần chạy gồm `seed_base` (= Base_fix, cùng a_p và a_n) và arm đang thử, nên so sánh công bằng.

```bash
cd $CODE
submit_chain() {   # submit_chain <ARM> <số job> [jobid-trước]
  local arm=$1 n=$2 prev=${3:-} j i
  for ((i = 0; i < n; i++)); do
    if [ -z "$prev" ]; then
      j=$(qsub -l walltime=03:00:00 -v ARM=$arm scripts/run_exp1.pbs)
    else
      j=$(qsub -l walltime=03:00:00 -W depend=afterany:$prev -v ARM=$arm scripts/run_exp1.pbs)
    fi
    prev=$j
  done
  echo "$prev"
}
LAST=$(submit_chain adaptive_pos_only 8)
LAST=$(submit_chain adaptive_neg_only 8 $LAST)
LAST=$(submit_chain adaptive_pos_neg 8 $LAST)
echo "job cuối: $LAST"
```

- `afterany` làm job sau chạy khi job trước kết thúc, kể cả bị hết walltime. Mọi công việc đã xong được
  bỏ qua, nên job thừa tốn rất ít SU: ngưỡng đã có kết quả được log `SKIP (đã xong)`.
- Số job mỗi arm (ở đây 8) chỉ là ước lượng. Đo thời gian thật từ log của job đầu, rồi tăng hoặc giảm.
- Nếu có lỗi thật (không phải hết giờ), mọi job trong chuỗi sẽ lỗi nhanh. Xem log, sửa, rồi `qdel` phần còn lại.

### Biến tuỳ chọn (qua `qsub -v`, phân tách bằng dấu phẩy)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `VALUES` | `0.25 0.5 0.75 0.9 1` | Ví dụ `VALUES="0.5 0.9"` (đặt trong dấu nháy) |
| `FIXED_AP` | `0.9` | a_p khi chỉ quét a_n |
| `FIXED_AN` | `0.25` | a_n khi chỉ quét a_p |
| `TEACHER_FRAMES` | `32` | Số frame của teacher |
| `PAIR` | unset | Một cặp Both dạng `a_pos:a_neg`, ví dụ `1:0.75`; nhận cả 0 |
| `PAIRS` | unset | Nhiều cặp Both phân tách bằng `;` |
| `DISTILL_METHODS` | unset | So sánh `kd_basic:video_opd` |
| `WORK` | `/scratch/hn98/$USER/runs/exp1` | Work-dir |
| `SEED_CHECKPOINT_EVERY_STEPS` | `5` | Lưu checkpoint train mỗi số bước optimizer |
| `SEED_EVAL_CHECKPOINT_EVERY` | `25` | Lưu tiến độ đánh giá mỗi số mẫu |

### So sánh distillation methods

Hai nhánh dùng cùng teacher `T_pos_tcd` và cùng `(a_pos, a_neg)`. `kd_basic` là baseline forward-KL
trên phân phối yes/no ở prompt. `video_opd` là bản thích nghi Video-OPD cho QA video nhị phân: student
sinh câu trả lời từ prompt của benchmark; teacher phải dự đoán đúng nhãn để qua reliability gate; trong
các rollout đó, TVDF giữ nhóm có reverse-KL trung bình theo token cao nhất. Rollout được dùng lại cho
cập nhật student và được lưu cùng checkpoint để resume nhất quán.

Đây là adaptation theo cơ chế, không phải reproduction cấu hình chính thức: dùng InternVL3.5/Qwen3 hiện
có, teacher tổng hợp `T_pos_tcd`, dữ liệu VideoHallucer và binary QA. `kd_basic` cũng là đối chứng cùng
teacher, không phải một phần của Video-OPD. SEED không nằm trong bảng so sánh này vì không có mã nguồn
tham chiếu để xác nhận implementation. OPSD-V là self-distillation cho mô hình sinh video khuếch tán;
objective của nó không tương thích trực tiếp với student VLM đang sinh câu trả lời QA, nên không gắn nhãn
phương pháp hiện tại là OPSD-V.

Chạy smoke trước:

```bash
qsub -l walltime=00:30:00 \
  -v "ARM=adaptive_pos_neg,PAIR=0.9:0.25,DISTILL_METHODS=kd_basic:video_opd,SMOKE_PAIRS=1,WORK=$SMOKE,TEACHER_FRAMES=32" \
  -N seed_distill_smoke \
  -o "$PWD/logs/seed_distill_smoke.out" \
  -e "$PWD/logs/seed_distill_smoke.err" \
  scripts/run_exp1.pbs
```

Smoke và full comparison đều chạy hai epoch: epoch đầu warm-up, epoch sau áp dụng TVDF. Như vậy smoke xác
minh cả rollout/loss lẫn reliability gate, xếp hạng TVDF và dùng lại rollout; KD cũng chạy hai epoch để so
sánh công bằng. Kết quả nằm dưới
`$WORK/cdt_distill_cache_f8_448/distillation_methods/runs/ap0.9_an0.25_tf32/`.

### Grid Both 2D

Mỗi cặp là một cấu hình riêng; `a_pos=0` tắt nhánh positive, `a_neg=0` tắt nhánh TCD-negative, `(0,0)` là
control không thêm nhánh nào. Với `{0, 0.25, 0.5, 0.75, 0.9, 1}`, grid có 36 cặp, gồm điểm như `(1,0.75)`.
Submit từng cặp để dễ resume:

```bash
PAIR=1:0.75
qsub -l walltime=02:00:00 \
  -v "ARM=adaptive_pos_neg,PAIR=$PAIR,WORK=$WORK,TEACHER_FRAMES=32" \
  -N "seed_both_${PAIR/:/_}" \
  -o "$PWD/logs/seed_both_${PAIR/:/_}.out" \
  -e "$PWD/logs/seed_both_${PAIR/:/_}.err" \
  scripts/run_exp1.pbs
```

Để xem grid và ghi record thiếu:

```bash
python scripts/collect_exp1.py --work "$WORK" \
  --both-pairs 0:0 0:0.25 0:0.5 0:0.75 0:0.9 0:1 \
  0.25:0 0.25:0.25 0.25:0.5 0.25:0.75 0.25:0.9 0.25:1 \
  0.5:0 0.5:0.25 0.5:0.5 0.5:0.75 0.5:0.9 0.5:1 \
  0.75:0 0.75:0.25 0.75:0.5 0.75:0.75 0.75:0.9 0.75:1 \
  0.9:0 0.9:0.25 0.9:0.5 0.9:0.75 0.9:0.9 0.9:1 \
  1:0 1:0.25 1:0.5 1:0.75 1:0.9 1:1
```

Method comparison summary:

```bash
python scripts/collect_distillation_methods.py --work "$WORK" --a-pos 0.9 --a-neg 0.25
```

### Chạy song song (tuỳ chọn, tốn thêm SU và dung lượng)

Mỗi arm một work-dir riêng, mỗi work-dir tự tính lại teacher. Dựng work-dir bằng cùng các bước ở mục 1.4 và 1.6,
rồi dùng chung mask foreground để khỏi tính lại (khoảng 14 phút mỗi work-dir):

```bash
SRC_FG=$WORK/cdt_distill_cache_f8_448/fg_81b255eaea
for arm in adaptive_pos_only adaptive_neg_only adaptive_pos_neg; do
  W=/scratch/hn98/$USER/runs/exp1_$arm
  mkdir -p $W/VideoHallucer $W/cdt_distill_cache_f8_448
  cp -rs $SRC_VH/temporal $SRC_VH/object_relation $W/VideoHallucer/
  prep_eh $W
  [ -e $W/cdt_distill_cache_f8_448/fg_81b255eaea ] || cp -rs $SRC_FG $W/cdt_distill_cache_f8_448/fg_81b255eaea
done
```

Submit mỗi arm với `WORK` riêng, ví dụ:

```bash
qsub -l walltime=03:00:00 -v ARM=adaptive_pos_only,WORK=/scratch/hn98/$USER/runs/exp1_adaptive_pos_only scripts/run_exp1.pbs
```

Thư mục `fg_81b255eaea` chỉ tồn tại sau khi một lần chạy đầy đủ trên `$WORK` đã xong bước mask.
Tên này là hash cấu hình mask (32 frame); nếu đổi `TEACHER_FRAMES` thì tên khác.

## 4. Theo dõi

```bash
qstat -u $USER            # Q = chờ, R = chạy, H = bị giữ (chờ job trước)
qstat -xu $USER | tail    # job đã xong (F)
tail -n 40 seed_exp1.o<jobid>
qstat -fx <jobid> | grep -E "Exit_status|resources_used|comment"
```

Mốc cần thấy trong log, theo thứ tự:

1. `student n_frames = 8 | teacher_frames = 32`
2. `HF_HUB_OFFLINE=1: dùng dữ liệu local ...` và `train 150 cặp | val 50 cặp | test 155 cặp`
3. `EventHallusion: ... câu / ... video | thiếu: 0`
4. `fg cache: ... | lưới (16, 16, 16)` (lần đầu mất khoảng 30 phút cho cả VideoHallucer và EventHallusion)
5. `null positive vs orig: max|Δ| = 0.00e+00`
6. Thanh `Shared teacher views TRAIN` (tính teacher 32 frame, chỉ một lần)
7. `[seed_base_s0] epoch 0: {...}` rồi `[<ARM>_s0] epoch 0: {...}`
8. Mỗi ngưỡng bắt đầu bằng `===== <ARM> | a_pos=... a_neg=... =====`, hoặc `SKIP (đã xong)` nếu đã có kết quả

Khi job tiếp tục sau khi bị ngắt, log có dòng `resume checkpoint: epoch=..., next_sample=...`.

`test 155 cặp` (không phải 300) là bình thường: tổng chỉ có 376 cặp và code loại các cặp trùng video với train/val.

## 5. Kết quả từng cấu hình

```
$WORK/cdt_distill_cache_f8_448/seed_component_ablation/runs/ap<a>_an<b>_tf32/
├── results.csv                    # q_acc, pair_acc, yes_rate, delta_pp, wrong_to_correct, correct_to_wrong
├── target_diagnostics_train.csv   # hệ số a_p, a_n thực tế theo từng mẫu train
└── <arm>_s0_predictions.csv       # margin theo từng câu hỏi
```

Các `split` trong `results.csv`: `VH_VAL`, `VH_TEST` (có `pair_acc`), `EH_ALL`, `eh_entire`, `eh_misleading`, `eh_mix`.
`wrong_to_correct` và `correct_to_wrong` đếm so với Qwen gốc (`base`), không phải so với `seed_base`.

Một ngưỡng xong khi `results.csv` của nó có cả dòng `seed_base_s0` và `<arm>_s0`:

```bash
grep -c "^adaptive_pos_only_s0," $WORK/cdt_distill_cache_f8_448/seed_component_ablation/runs/*/results.csv
```

## 6. Gộp bảng báo cáo

```bash
source /scratch/hn98/$USER/venv/bin/activate
python $CODE/scripts/collect_exp1.py --work $WORK
```

Bảng in ra theo thứ tự cột: **VH pair acc, VH Q acc, EH acc**, tiếp theo là số câu đổi sai thành đúng và đúng thành sai
(`VH w2c`, `VH c2w`, `EH w2c`, `EH c2w`). Mỗi ngưỡng có hai dòng, `Base_fix` và `Adaptive`, để so Acc(adaptive) > Acc(Base_fix).
Bảng cũng được lưu ở `$WORK/exp1_table.csv`.

Dùng tập validation thay cho test: thêm `--vh-split VH_VAL`.
Thiếu kết quả của ngưỡng nào thì dòng đó ghi `(missing results)`.

## 7. Resume và chạy lại

| Giai đoạn | Mức resume |
|---|---|
| Mask foreground | từng video |
| Logit teacher (32 frame) | từng mẫu TRAIN |
| Điểm `base` (Qwen gốc) | không; chỉ lưu khi xong cả tập (khoảng chục phút) |
| Train student | mỗi 5 bước optimizer |
| Đánh giá student | mỗi 25 mẫu |
| Arm đã xong | cache adapter và margin |

- Resume cần **đúng cùng `ARM`, `VALUES`, `WORK`** và **cùng mã nguồn**. Sửa file `.py` trong `src/` làm hash mã
  nguồn đổi, nên cache logit teacher và student cũ không được nhận. Mask foreground không bị ảnh hưởng.
- Nếu stage foreground báo lỗi `torch.load` sau một lần bị kill, file mask mới nhất có thể hỏng. Xoá nó rồi chạy lại:

```bash
ls -t $WORK/cdt_distill_cache_f8_448/fg_*/*.pt | head -1
```

- Không xoá `fg_*` và `seed_component_ablation/views_*` khi còn định chạy thêm: tính lại tốn nhiều giờ GPU.

## 8. Xử lý lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `Network is unreachable` ở stage `data.eventhallusion` | Work-dir thiếu `EventHallusion/questions/*.json`. Chạy `prep_eh <work-dir>` |
| `EventHallusion thiếu N video` | Thiếu video trong `<work-dir>/EventHallusion/`. Tải bản đầy đủ (mục 1.5) rồi `prep_eh` |
| `qsub: Job rejected ... queue` | Project không có quyền queue. Thử `-P jp09` trong `.pbs`, giữ `storage=scratch/hn98+scratch/jp09` |
| Job `Q` rất lâu | Hàng đợi H200 kín (120 GPU). Xem `qstat -Q gpuhopper`; có thể đổi sang `dgxa100` (`ncpus=16`, `mem=120GB`) |
| `/scratch/hn98: No such file` trong job | Thiếu `scratch/hn98` trong dòng `#PBS -l storage=` |
| Lỗi tải model khi chạy | Node GPU không có Internet. Cần model nằm sẵn trong `$HF_HOME`, kèm `HF_HUB_OFFLINE=1` |
| `Disk quota exceeded` | Home đầy (10 GB). Kiểm tra `quota -s`; chuyển cache sang `/scratch` |
| Hết walltime | Bình thường với job 3 giờ; job kế tiếp trong chuỗi tiếp tục từ checkpoint |
| Job sau bị `H` mãi | Phụ thuộc không thoả. `qdel` job bị giữ và submit lại |
| `git pull` không chạy được trong job | Node GPU không có Internet; chỉ `git pull` ở login node |

Dừng một job: `qdel <jobid>`. Dừng cả chuỗi: `qdel $(qselect -u $USER -N seed_exp1)`.

## 9. Dọn dẹp

```bash
lquota
du -sh $WORK/cdt_distill_cache_f8_448/*
```
