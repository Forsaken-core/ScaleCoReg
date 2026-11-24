from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_EPS = np.finfo(float).eps


def event_union(values):
    values = sorted(float(x) for x in values if 0 < x < 1 )
    groups = []
    for x in values:
        if groups and abs(x - groups[-1][-1]) <= 32 * _EPS:
            groups[-1].append(x)
        else:
            groups.append([x])
    return np.array([sum(g) / len(g) for g in groups])


def crossings(a, b, c, d, p):
    if p == 2:
        v0 = np.dot(a, a) - np.dot(c, c)
        v1 = np.dot(b, b) - np.dot(d, d)
        denom = v0 - v1
        return [] if denom == 0 else [v0 / denom]

    starts = np.r_[a, c]
    slopes = np.r_[b - a, d - c]
    nz = slopes != 0
    cuts = np.r_[0., event_union(-starts[nz] / slopes[nz]), 1.]
    roots = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = (lo + hi) / 2
        sa, sc = np.sign(a + mid * (b - a)), np.sign(c + mid * (d - c))
        def poly(x, v, sign):
            return np.array([np.sum(sign * x**3), np.sum(sign * 3*x*x*v),
                             np.sum(sign * 3*x*v*v), np.sum(sign * v**3)])
        coef = poly(a, b-a, sa) - poly(c, d-c, sc)
        scale = np.max(np.abs(coef))
        if scale == 0:
            continue
        while len(coef) > 1 and abs(coef[-1]) <= 16 * _EPS * scale:
            coef = coef[:-1]
        for root in np.polynomial.polynomial.polyroots(coef / scale):
            if abs(root.imag) <= 128 * _EPS * max(1., abs(root.real)):
                x = float(root.real)
                if lo - 32*_EPS <= x <= hi + 32*_EPS:
                    roots.append(x)
    return roots


@dataclass
class Timeline:
    initial: tuple
    events: list


class ScaleSelector:
    def __init__(self, features, labels, support_ids, anchor_ids, head, k=3, p=2):
        self.z = np.asarray(features, dtype=float).copy()
        self.y = np.asarray(labels, dtype=float).reshape(-1).copy()
        self.ids = np.asarray(support_ids, dtype=np.int64).copy()
        w = np.asarray(head, dtype=float).reshape(-1)
        norm = np.linalg.norm(w)
        self.unit = w / norm
        self.k, self.p = int(k), int(p)
        lookup = {int(x): i for i, x in enumerate(self.ids)}
        self.anchors = [lookup[int(x)] for x in anchor_ids]
        self.cache = {}

    def components(self, delta):
        parallel = np.outer(delta @ self.unit, self.unit)
        return parallel, delta - parallel

    def distance(self, a, b, r):
        if self.p == 2:
            return (1-r)*np.sum(a*a, axis=1) + r*np.sum(b*b, axis=1)
        return np.sum(np.abs((1-r)*a + r*b)**3, axis=1)

    def order(self, a, b, rows, ids, r, ties=False):
        dist = self.distance(a, b, r)
        order = np.argsort(dist, kind="stable")
        if ties:

            out, start = [], 0
            while start < len(order):
                end = start + 1
                tol = 128*_EPS*max(1., abs(dist[order[start]]))
                while end < len(order) and dist[order[end]]-dist[order[start]] <= tol:
                    end += 1
                group = order[start:end]
                out.extend(group[np.argsort(ids[group], kind="stable")])
                start = end
            order = np.asarray(out)
        return tuple(int(x) for x in rows[order[:self.k]])

    def background(self, anchor):
        if anchor in self.cache:
            return self.cache[anchor]
        rows = np.flatnonzero(self.ids != self.ids[anchor])
        a, b = self.components(self.z[rows] - self.z[anchor])
        roots = []
        for i in range(len(rows)):
            for j in range(i):
                roots.extend(crossings(a[i], b[i], a[j], b[j], self.p))
        roots = event_union(roots)
        first = roots[0] if len(roots) else 1.
        current = initial = self.order(a, b, rows, self.ids[rows], first/2)
        events = []
        for i, r in enumerate(roots):
            at = self.order(a, b, rows, self.ids[rows], r, ties=True)
            nxt = roots[i+1] if i+1 < len(roots) else 1.
            right = self.order(a, b, rows, self.ids[rows], (r+nxt)/2)
            if at != current or right != current:
                events.append((float(r), at, right))
            current = right
        timeline = Timeline(initial, events)
        self.cache[anchor] = timeline
        return timeline

    def insertion(self, anchor, feature, candidate_id):
        bg = self.background(anchor)
        ca, cb = self.components(np.asarray(feature).reshape(1, -1)-self.z[anchor])
        roots = [r for r, _, _ in bg.events]


        pair_roots = {}
        old, lo = bg.initial, 0.
        boundaries = bg.events + [(1., (), ())]
        for hi, _, right in boundaries:
            for row in old:
                if row not in pair_roots:
                    a, b = self.components((self.z[row]-self.z[anchor]).reshape(1,-1))
                    pair_roots[row] = crossings(a[0],b[0],ca[0],cb[0],self.p)
                roots.extend(r for r in pair_roots[row]
                             if lo-32*_EPS <= r <= hi+32*_EPS)
            old, lo = right, hi
        roots = event_union(roots)
        newrow = len(self.z)
        def insert(old, r, ties=False):
            rr = np.array(list(old)+[newrow])
            zz = np.vstack([self.z[list(old)], feature])
            aa, bb = self.components(zz-self.z[anchor])
            ii = np.r_[self.ids[list(old)], candidate_id]
            return self.order(aa, bb, rr, ii, r, ties)
        def gain(old, new):
            if newrow not in new:
                return np.zeros(3), 0
            baseline = float(self.y[list(old)].mean())
            q = float(self.y[old[-1]])
            error = float(self.y[anchor])-baseline
            k = self.k
            return np.array([-2*error*q/k-q*q/(k*k),
                             2*error/k+2*q/(k*k), -1/(k*k)]), 1
        first = roots[0] if len(roots) else 1.
        old = bg.initial
        new = initial = insert(old, first/2)
        gain_initial = gain(old, new)
        gain_events, new_events = [], []
        bi = 0
        last_gain = gain_initial
        for i, r in enumerate(roots):
            atold = rightold = old
            if bi < len(bg.events) and abs(bg.events[bi][0]-r) <= 32*_EPS:
                _, atold, rightold = bg.events[bi]
                bi += 1
            nxt = roots[i+1] if i+1 < len(roots) else 1.
            atnew = insert(atold, r, True)
            rightnew = insert(rightold, (r+nxt)/2)
            atgain, rightgain = gain(atold, atnew), gain(rightold, rightnew)
            if (not np.array_equal(atgain[0], last_gain[0]) or atgain[1] != last_gain[1]
                or not np.array_equal(rightgain[0], last_gain[0]) or rightgain[1] != last_gain[1]):
                gain_events.append((float(r), atgain, rightgain))
            if atnew != new or rightnew != new:
                new_events.append((float(r), atnew, rightnew))
            old, new, last_gain = rightold, rightnew, rightgain
        return gain_initial, gain_events, Timeline(initial, new_events)

    def evaluate(self, feature, donor_label, recipient_label, candidate_id):
        feature = np.asarray(feature, dtype=float).reshape(-1)
        ta, tb = float(donor_label), float(recipient_label)


        total = np.zeros(3)
        count = 0
        for anchor in self.anchors:
            rows = np.flatnonzero(self.ids != self.ids[anchor])
            a,b = self.components(self.z[rows]-self.z[anchor])
            old = self.order(a,b,rows,self.ids[rows],.5,True)
            zz = np.vstack([self.z[list(old)],feature])
            aa,bb = self.components(zz-self.z[anchor])
            new = self.order(aa,bb,np.r_[list(old),len(self.z)],
                             np.r_[self.ids[list(old)],candidate_id],.5,True)
            if len(self.z) in new:
                baseline = self.y[list(old)].mean()
                q = self.y[old[-1]]
                e = self.y[anchor]-baseline
                k = self.k
                total += [-2*e*q/k-q*q/(k*k),2*e/k+2*q/(k*k),-1/(k*k)]
                count += 1
        if count:
            c,b,a = total/count
            fa = c+b*ta+a*ta*ta
            if fa <= 0:
                return {"accepted": False, "score": 0.}
        profiles, traces = [], {}
        for anchor in self.anchors:
            g0, events, trace = self.insertion(anchor, feature, candidate_id)
            profiles.append((g0, events))
            traces[anchor] = trace


        r_min, r_max = (.2, .8) if self.p == 2 else (1/3, 2/3)
        roots = event_union([r_min, r_max] +
                            [r for _, ev in profiles for r, _, _ in ev])
        states = [g0 for g0, _ in profiles]
        total = sum((x[0] for x in states), np.zeros(3))
        count = sum(x[1] for x in states)
        cursors = [0]*len(profiles)
        low, high, aa, ab = -np.inf, np.inf, np.inf, np.inf
        active = 0
        def check(r):
            nonlocal low, high, aa, ab, active
            if r < r_min or r > r_max or count == 0:
                return None
            active += 1
            c, b, a = total/count
            fa, fb = c+b*ta+a*ta*ta, c+b*tb+a*tb*tb
            aa, ab = min(aa, fa), min(ab, fb)
            tau = -b/(2*a)
            peak = c+b*tau+a*tau*tau
            if fa <= 0:
                return {"accepted": False, "score": 0.}
            radius = self.k*np.sqrt(peak)
            low, high = max(low, tau-radius), min(high, tau+radius)
            return None
        result = check(roots[0]/2)
        if result is None:
            for root_idx, r in enumerate(roots):
                changed = []
                for i, (_, ev) in enumerate(profiles):
                    j = cursors[i]
                    if j < len(ev) and abs(ev[j][0]-r) <= 32*_EPS:
                        total -= states[i][0]
                        count -= states[i][1]
                        states[i] = ev[j][1]
                        total += states[i][0]
                        count += states[i][1]
                        changed.append(i)
                result = check(r)
                if result is not None:
                    break
                for i in changed:
                    ev = profiles[i][1]
                    j = cursors[i]
                    total -= states[i][0]
                    count -= states[i][1]
                    states[i] = ev[j][2]
                    total += states[i][0]
                    count += states[i][1]
                    cursors[i] += 1
                next_r = roots[root_idx+1] if root_idx+1 < len(roots) else 1.
                result = check((r+next_r)/2)
                if result is not None:
                    break
        if result is not None:
            return result
        if not active or not low < ta < high:
            return {"accepted": False, "score": 0.}
        advantage = aa-ab
        score = aa+min(aa, max(0., advantage))
        return {"accepted": True,
                "interval": [float(low), float(high)], "A_donor": float(aa),
                "A_recipient": float(ab), "advantage": float(advantage),
                "score": float(score), "_traces": traces}

    def append(self, feature, label, candidate_id, result):
        self.z = np.vstack([self.z, np.asarray(feature, dtype=float)])
        self.y = np.r_[self.y, float(label)]
        self.ids = np.r_[self.ids, int(candidate_id)]
        self.cache = result["_traces"]
