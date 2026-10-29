"""data: eventhallusion stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def run(runtime):
    import urllib.request, zipfile
    runtime.urllib = urllib
    runtime.zipfile = zipfile
    from decord import VideoReader, cpu as _dcpu
    runtime.VideoReader = VideoReader
    runtime._dcpu = _dcpu
    runtime.EH = []
    if runtime.CFG['eh']['enable']:
        runtime.E = runtime.CFG['eh']
        runtime.eroot = runtime.E['root']
        runtime.os.makedirs(f'{runtime.eroot}/questions', exist_ok=True)
        for runtime.sp in runtime.E['splits']:
            runtime.fp = f'{runtime.eroot}/questions/{runtime.sp}_questions.json'
            if not runtime.os.path.exists(runtime.fp):
                runtime.urllib.request.urlretrieve(f"https://raw.githubusercontent.com/Stevetich/EventHallusion/{runtime.E['q_commit']}/questions/{runtime.sp}_questions.json", runtime.fp)
        runtime.VEXT = ('.mp4', '.mkv', '.webm', '.avi', '.mov')

        def _index():
            by_dir = {}
            for p_ in runtime.glob.glob(f'{runtime.eroot}/**/*.*', recursive=True):
                if '__MACOSX' in p_ or not p_.lower().endswith(runtime.VEXT):
                    continue
                by_dir.setdefault(runtime.os.path.basename(runtime.os.path.dirname(p_)).lower(), {})[runtime.os.path.splitext(runtime.os.path.basename(p_))[0]] = p_
            return by_dir
        runtime._index = _index
        runtime.vids_by_dir = runtime._index()
        if not runtime.vids_by_dir:
            runtime.zp = f'{runtime.eroot}/videos.zip'
            if not runtime.os.path.exists(runtime.zp):
                import gdown
                runtime.gdown = gdown
                runtime.gdown.download(id=runtime.E['gdrive_id'], output=runtime.zp, quiet=False)
            with runtime.zipfile.ZipFile(runtime.zp) as runtime.z:
                runtime.z.extractall(runtime.eroot)
            runtime.vids_by_dir = runtime._index()
        print('Thư mục video:', {k: len(v) for k, v in runtime.vids_by_dir.items()})
        runtime.ALL_EH_FILES = {p_ for d in runtime.vids_by_dir.values() for p_ in d.values()}
        runtime.SUBDIRS = {'entire': ['entire'], 'mix': ['interleave', 'mix'], 'misleading': ['misleading']}

        def _cands(vid, sp):
            pre, _, num = vid.partition('_')
            n = str(int(num)) if num.isdigit() else num
            out = [vid, f'{pre}_{n}', f'{pre}{n}', num, n]
            if sp == 'mix':
                out += [f'interleave_{num}', f'interleave_{n}', f'interleave{n}']
            return list(dict.fromkeys(out))
        runtime._cands = _cands

        def resolve(vid, sp):
            for d in runtime.SUBDIRS.get(sp, [sp]):
                files = runtime.vids_by_dir.get(d, {})
                for c in runtime._cands(vid, sp):
                    if c in files:
                        return (files[c], d)
            return (None, None)
        runtime.resolve = resolve

        def dur_sec(path):
            vr = runtime.VideoReader(path, ctx=runtime._dcpu(0))
            return len(vr) / (vr.get_avg_fps() or 30.0)
        runtime.dur_sec = dur_sec

        def len_sec(value):
            """Đổi length dạng số giây hoặc chuỗi HH:MM:SS/MM:SS thành giây."""
            if isinstance(value, (int, float, runtime.np.integer, runtime.np.floating)):
                return float(value)
            value = str(value).strip()
            try:
                return float(value)
            except ValueError:
                pass
            parts = [float(x.strip()) for x in value.split(':')]
            if not 1 <= len(parts) <= 3:
                raise ValueError(f'Định dạng length không hợp lệ: {value!r}')
            return sum((part * 60 ** power for power, part in enumerate(reversed(parts))))
        runtime.len_sec = len_sec
        runtime.miss, runtime.used, runtime.chk = ([], set(), [])
        for runtime.sp in runtime.E['splits']:
            for runtime.v in runtime.json.load(open(f'{runtime.eroot}/questions/{runtime.sp}_questions.json')):
                runtime.path, runtime.folder = runtime.resolve(runtime.v['id'], runtime.sp)
                if runtime.path is None:
                    runtime.miss.append(runtime.v['id'])
                    continue
                runtime.used.add(runtime.path)
                if runtime.v.get('length'):
                    runtime.chk.append(dict(split=runtime.sp, id=runtime.v['id'], folder=runtime.folder, json_s=runtime.len_sec(runtime.v['length']), video_s=runtime.dur_sec(runtime.path)))
                for runtime.qi, runtime.q in enumerate(runtime.v['questions']):
                    runtime.EH.append(dict(subset=f'eh_{runtime.sp}', pair_id=f"eh/{runtime.v['id']}/{runtime.qi}", role='eh', video=runtime.path, question=runtime.q['question'], answer=runtime.q['answer'].strip().strip('.').lower()))
        assert all((i['answer'] in ('yes', 'no') for i in runtime.EH))
        runtime.chk = runtime.pd.DataFrame(runtime.chk)
        if len(runtime.chk):
            runtime.chk['ok'] = (runtime.chk.json_s - runtime.chk.video_s).abs() <= 1.5
            print("Khớp thời lượng video với 'length' trong json (sai số ≤ 1.5 s):")
            runtime.display(runtime.chk.groupby(['split', 'folder']).agg(n=('ok', 'size'), ok_rate=('ok', 'mean')).round(3))
            if (~runtime.chk.ok).any():
                print('Ví dụ lệch:')
                runtime.display(runtime.chk[~runtime.chk.ok].head(8).round(2))
            for runtime.sp_, runtime.r_ in runtime.chk.groupby('split').ok.mean().items():
                if runtime.r_ < 0.9:
                    raise RuntimeError(f'Split {runtime.sp_}: chỉ {runtime.r_:.0%} video khớp thời lượng → ánh xạ id ↔ file có thể sai.')
        if runtime.miss:
            runtime.unused = sorted((runtime.os.path.relpath(p_, runtime.eroot) for p_ in runtime.ALL_EH_FILES - runtime.used))
            print('Thư mục chứa video:', sorted({runtime.os.path.relpath(runtime.os.path.dirname(p_), runtime.eroot) for p_ in runtime.ALL_EH_FILES}))
            print(f'File chưa khớp id nào: {len(runtime.unused)}', runtime.unused[:10])
            if not runtime.E['allow_missing']:
                raise RuntimeError(f'EventHallusion thiếu {len(runtime.miss)} video (vd {runtime.miss[:5]}). Xem 2 dòng in phía trên.')
        runtime.ALL_VIDEOS = sorted(set(runtime.ALL_VIDEOS) | {i['video'] for i in runtime.EH})
        print(f"EventHallusion: {len(runtime.EH)} câu / {len({i['video'] for i in runtime.EH})} video | thiếu: {len(runtime.miss)}")
        runtime.display(runtime.pd.DataFrame([dict(split=s, n=sum((i['subset'] == s for i in runtime.EH)), gt_yes=runtime.np.mean([i['answer'] == 'yes' for i in runtime.EH if i['subset'] == s])) for s in sorted({i['subset'] for i in runtime.EH})]).round(3))
    runtime.EH_SPLITS = [s for s in ['eh_entire', 'eh_misleading', 'eh_mix'] if any((i['subset'] == s for i in runtime.EH))]
