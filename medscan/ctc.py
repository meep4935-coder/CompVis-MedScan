"""
Re-read a field under rules, using the recogniser's per-step probabilities (CTC).

Coming from MLPs: the recogniser outputs, for every time step t across the line image,
a softmax over characters (plus "blank"). A text is read by choosing one symbol per step,
then merging repeats and dropping blanks: "--TT-R-EE-M" -> "TREM". The usual greedy read
takes the top symbol at every step. Here we instead search for the single most probable
path whose merged text MATCHES A PATTERN, e.g. RAMQ = 4 letters, optional space,
4 digits, optional space, 4 digits. This is a Viterbi search (dynamic programming over
time steps x pattern positions), the same idea used in speech recognition with grammars.

Why it helps: if a step says "3: 0.45, J: 0.40" in a letter position, the pattern picks J.
If the greedy read dropped a digit, the pattern forces 8 digits and finds where the model
had a weaker second guess for the missing one.
"""

import math
import string

FLOOR = math.log(1e-4)  # probability given to characters not in a step's top-k
UPPER = set(string.ascii_uppercase)
LETTERS = UPPER | set(string.ascii_lowercase)
DIGITS = set(string.digits)
SEP = {" ", "-"}
DATE_SEP = {"-", "/", ".", " "}


def logp(frame, ch):
    for c, p in frame:
        if c == ch:
            return math.log(max(p, 1e-6))
    return FLOOR


def spans(frames):
    """Greedy read with the time steps of each character: [(char, t_first, t_last)]."""
    out, prev = [], ""
    for t, f in enumerate(frames):
        c = f[0][0] if f else ""
        if c and c == prev:
            out[-1] = (c, out[-1][1], t)
        elif c:
            out.append((c, t, t))
        prev = c
    return out


def window(frames, sp, i, j):
    """Time steps covering greedy characters i..j (inclusive), up to the neighbours."""
    t0 = sp[i - 1][2] + 1 if i > 0 else 0
    t1 = sp[j + 1][1] - 1 if j + 1 < len(sp) else len(frames) - 1
    return t0, t1


def decode(frames, pattern, t0=0, t1=None):
    """Best path over frames[t0..t1] whose text fits `pattern`, a list of (allowed_chars,
    optional) positions. Returns (text, total log-prob, steps) or (None, -inf, steps)."""
    t1 = len(frames) - 1 if t1 is None else t1
    L = len(pattern)
    # positions reachable right after position k (skipping optional ones)
    nxt = []
    for k in range(-1, L):
        reach, m = [], k + 1
        while m < L:
            reach.append(m)
            if not pattern[m][1]:
                break
            m += 1
        nxt.append(reach)
    final = {k for k in range(-1, L) if all(pattern[m][1] for m in range(k + 1, L))}

    # state: (k, ch) = last emitted char ch at pattern position k; ch "" = blank after k
    best = {(-1, ""): (0.0, None)}
    history = []
    for t in range(t0, t1 + 1):
        f = frames[t]
        cand = {c for c, _ in f}
        new = {}

        def push(state, score, back):
            if state not in new or score > new[state][0]:
                new[state] = (score, back)

        lb = logp(f, "")
        for (k, ch), (s, _) in best.items():
            push((k, ""), s + lb, (k, ch))                    # blank
            if ch:
                push((k, ch), s + logp(f, ch), (k, ch))      # same char continues
            for m in nxt[k + 1]:
                allowed = pattern[m][0]
                options = (cand & allowed) or {min(allowed)}  # unseen chars share the floor
                for c in options:
                    if c == ch:
                        continue  # a doubled char needs a blank in between
                    push((m, c), s + logp(f, c), (k, ch))
        history.append(new)
        best = new

    ends = [(s, st) for st, (s, _) in best.items() if st[0] in final]
    if not ends:
        return None, -math.inf, len(history)
    score, st = max(ends)
    # backtrack: collect the char at each step where the state's position advanced
    text, steps = [], len(history)
    for t in range(steps - 1, -1, -1):
        s, back = history[t][st]
        if st[1] and (back[0] != st[0] or back[1] != st[1]):
            text.append(st[1])
        st = back
    return "".join(reversed(text)), score, steps


def pos(chars, optional=False):
    return (frozenset(chars), optional)


RAMQ = [pos(LETTERS)] * 4 + [pos(SEP, True)] + [pos(DIGITS)] * 4 + [pos(SEP, True)] + [pos(DIGITS)] * 4
DATE_YMD = [pos(DIGITS)] * 4 + [pos(DATE_SEP)] + [pos(DIGITS)] * 2 + [pos(DATE_SEP)] + [pos(DIGITS)] * 2
DATE_DMY = [pos(DIGITS)] * 2 + [pos(DATE_SEP)] + [pos(DIGITS)] * 2 + [pos(DATE_SEP)] + [pos(DIGITS)] * 4


def ramq_fixed(letters):
    """Pattern with the 4 letters fixed (to score 'is this RAMQ GIRJ rather than GIR3?')."""
    lp = [pos({c, c.lower()}) for c in letters]
    return lp + RAMQ[4:]
