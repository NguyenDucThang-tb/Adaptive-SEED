"""Video-MME accuracy and paired, video-clustered retention statistics."""
from collections import defaultdict
import math
import random


def summarize(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups['overall', 'all'].append(row)
        for field in ('duration', 'domain', 'sub_category', 'task_type'):
            groups[field, row[field]].append(row)
    return [dict(group=group, value=value, n=len(rows),
                 accuracy=sum(r['prediction'] == r['answer'] for r in rows) / len(rows),
                 invalid=sum(r['prediction'] not in ('A', 'B', 'C', 'D') for r in rows))
            for (group, value), rows in sorted(groups.items())]


def compare(base, student, epsilon_pp=1.0, iterations=2000, seed=42):
    """One-sided 95% lower bound for a predeclared non-inferiority margin.

    Questions from a video are resampled together, preserving their correlation.
    A lower bound above -epsilon supports non-inferiority, not equivalence.
    """
    if not math.isfinite(epsilon_pp) or epsilon_pp < 0 or iterations < 100:
        raise ValueError('Require a nonnegative finite margin and >=100 bootstrap iterations.')
    if len({r['question_id'] for r in base}) != len(base) or len({r['question_id'] for r in student}) != len(student):
        raise ValueError('Duplicate question IDs in predictions.')
    original = {r['question_id']: r for r in base}
    updated = {r['question_id']: r for r in student}
    if not original or original.keys() != updated.keys():
        raise ValueError('Base/student must evaluate the same nonempty question set.')
    clusters = defaultdict(list)
    wrong_to_correct = correct_to_wrong = 0
    for key in sorted(original):
        b, s = original[key], updated[key]
        for field in ('videoID', 'answer', 'duration', 'question', 'options'):
            if b[field] != s[field]:
                raise ValueError(f'Base/student disagree on {field} for {key}')
        bc, sc = b['prediction'] == b['answer'], s['prediction'] == s['answer']
        clusters[b['videoID']].append(int(sc) - int(bc))
        wrong_to_correct += not bc and sc
        correct_to_wrong += bc and not sc
    videos = sorted(clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [clusters[videos[rng.randrange(len(videos))]] for _ in videos]
        samples.append(100 * sum(sum(c) for c in draw) / sum(map(len, draw)))
    samples.sort()
    lower = samples[int((iterations - 1) * 0.05)]
    base_acc = sum(r['prediction'] == r['answer'] for r in base) / len(base)
    student_acc = sum(r['prediction'] == r['answer'] for r in student) / len(student)
    return dict(n=len(base), n_videos=len(videos), base_accuracy=base_acc,
                student_accuracy=student_acc, delta_pp=100 * (student_acc - base_acc),
                lower_95_one_sided_pp=lower, epsilon_pp=epsilon_pp,
                noninferior=lower > -epsilon_pp, wrong_to_correct=wrong_to_correct,
                correct_to_wrong=correct_to_wrong, bootstrap_iterations=iterations, seed=seed)
