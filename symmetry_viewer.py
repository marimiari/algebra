import argparse
import itertools
import math
import os
import sys

import numpy as np

#алгебра операций симметрии

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rot(axis, angle):
    n = unit(axis)
    K = np.array([[0, -n[2], n[1]],
                  [n[2], 0, -n[0]],
                  [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def refl(normal):
    n = unit(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)


class Op:
    def __init__(self, kind, sym, desc, vec=None, angle=0.0, rot_sym=None):
        self.kind = kind
        self.sym = sym
        self.desc = desc
        self.vec = None if vec is None else unit(vec)
        self.angle = angle
        self.rot_sym = rot_sym
        self.where = ''

    def at(self, where):
        self.where = where
        return self

    @property
    def formula(self):
        return coord_formula(self.M)

    @property
    def M(self):
        if self.kind == 'E':
            return np.eye(3)
        if self.kind == 'C':
            return rot(self.vec, self.angle)
        if self.kind == 's':
            return refl(self.vec)
        if self.kind == 'i':
            return -np.eye(3)
        if self.kind == 'S':
            return refl(self.vec) @ rot(self.vec, self.angle)
        raise ValueError(self.kind)

    def apply(self, X):
        return X @ self.M.T


Z = np.array([0.0, 0.0, 1.0])
DEG = math.pi / 180

_COEF = [(1.0, ''), (0.5, '½'), (math.sqrt(3) / 2, '(√3/2)'), (1 / math.sqrt(2), '(1/√2)')]


def coord_formula(M):
    rows = []
    for r in M:
        terms = []
        for c, v in zip(r, 'xyz'):
            if abs(c) < 1e-9:
                continue
            name = None
            for val, txt in _COEF:
                if abs(abs(c) - val) < 1e-6:
                    name = txt
            if name is None:
                name = f'{abs(c):.3f}'
            sign = '−' if c < 0 else '+'
            terms.append((sign, name + v))
        if not terms:
            rows.append('0')
            continue
        out = ('−' if terms[0][0] == '−' else '') + terms[0][1]
        for sg, t in terms[1:]:
            out += f' {sg} {t}'
        rows.append(out)
    return '(x, y, z) → (' + ', '.join(rows) + ')'


def E():
    return Op('E', r'$E$', 'тождественная операция')


def Cn(n, k=1, axis=Z, sym=None, desc=None):
    ang = 2 * math.pi * k / n
    if sym is None:
        sym = rf'$C_{{{n}}}$' if k == 1 else rf'$C_{{{n}}}^{{{k}}}$'
    if desc is None:
        desc = f'поворот на {round(360 * k / n)}°'
    return Op('C', sym, desc, axis, ang)


def Sn(n, k=1, axis=Z, sym=None):
    ang = (2 * math.pi * k / n) % (2 * math.pi)
    if sym is None:
        sym = rf'$S_{{{n}}}$' if k == 1 else rf'$S_{{{n}}}^{{{k}}}$'
    deg = round(math.degrees(ang))
    rot_sym = rf'$C_{{{n}}}$' if (k % n) == 1 else rf'$C_{{{n}}}^{{{k % n}}}$'
    return Op('S', sym, f'поворот на {deg}° + отражение σh', axis, ang,
              rot_sym=rot_sym)


def sigma(normal, sym=r'$\sigma$', desc='отражение в плоскости'):
    return Op('s', sym, desc, normal)


def inv():
    return Op('i', r'$i$', 'инверсия в центре')


# модели молекул

COVALENT = {'H': 0.31, 'B': 0.84, 'C': 0.76, 'O': 0.66, 'Cl': 1.02}
COLORS = {'H': '#EDEFF2', 'B': '#F4A6A6', 'C': '#5C6370', 'O': '#E8473F',
          'Cl': '#4CC16A'}
SIZES = {'H': 130, 'B': 340, 'C': 300, 'O': 320, 'Cl': 400}


class Molecule:
    def __init__(self, key, name, formula, group, elems, X, ops, absent,
                 elements, labels=None, size=1.0, view=(25, 35),
                 note='', chiral_view=None):
        self.key = key
        self.name = name
        self.formula = formula
        self.group = group
        self.elems = list(elems)
        self.X = np.asarray(X, dtype=float)
        self.ops = ops
        self.absent = absent
        self.elements = elements
        self.labels = (list(range(len(elems))) if labels is None
                       else list(labels))
        self.size = size
        self.view = view
        self.note = note
        self.chiral_view = chiral_view
        self.bonds = find_bonds(self.elems, self.X)

    @property
    def extent(self):
        return float(np.max(np.linalg.norm(self.X, axis=1)))


def find_bonds(elems, X, factor=1.18):
    bonds = []
    for i, j in itertools.combinations(range(len(elems)), 2):
        d = np.linalg.norm(X[i] - X[j])
        if d < factor * (COVALENT[elems[i]] + COVALENT[elems[j]]):
            bonds.append((i, j))
    return bonds


def boric_acid():
    rBO, rOH, aBOH = 1.367, 0.970, 114.0 * DEG
    elems, X = ['B'], [[0, 0, 0]]
    O, H = [], []
    for k in range(3):
        t = 2 * math.pi * k / 3
        u = np.array([math.cos(t), math.sin(t), 0])
        tv = np.array([-math.sin(t), math.cos(t), 0])
        o = rBO * u
        h = o + rOH * (-math.cos(aBOH) * u + math.sin(aBOH) * tv)
        O.append(o)
        H.append(h)
    elems += ['O'] * 3 + ['H'] * 3
    X += O + H
    zax = 'ось z: перпендикулярна плоскости молекулы, проходит через атом B (1)'
    ops = [E().at('элемент не нужен — все атомы остаются на месте'),
           Cn(3, 1).at(zax), Cn(3, 2).at(zax),
           sigma(Z, r'$\sigma_h$', 'отражение в плоскости молекулы').at(
               'плоскость xy — сама плоскость молекулы, в ней лежат все 7 атомов'),
           Sn(3, 1).at(zax + '; плоскость σh — xy'), Sn(3, 5).at(zax + '; плоскость σh — xy')]
    absent = [Cn(2, 1, axis=[1, 0, 0], sym=r"$C_2'$",
                 desc='ось C2 вдоль связи B–O (ось x)'),
              sigma([0, 1, 0], r'$\sigma_v$', 'плоскость xz (через ось z и связь B–O)'),
              inv()]
    elements = [('axis', Z, r'$C_3,\ S_3$ (ось z)'),
                ('plane', Z, r'$\sigma_h$ (плоскость xy)')]
    return Molecule('boric', 'Борная кислота', 'B(OH)$_3$', r'$C_{3h}$',
                    elems, X, ops, absent, elements, size=1.0, view=(38, -60),
                    note='плоская, H повёрнуты «пропеллером»')


TPB_TWIST = 32.0


def triphenylborane(twist=TPB_TWIST):
    rBC, rAr, rCH = 1.57, 1.39, 1.08
    u = np.array([1.0, 0, 0])
    t = np.array([0, 1.0, 0])
    w = math.cos(twist * DEG) * t + math.sin(twist * DEG) * Z
    ctr = (rBC + rAr) * u
    ringC, ringH = [], []
    for m in range(6):
        al = m * math.pi / 3
        r = math.cos(al) * (-u) + math.sin(al) * w
        ringC.append(ctr + rAr * r)
        if m:
            ringH.append(ctr + (rAr + rCH) * r)
    gen = [np.eye(3), Cn(3, 1).M, Cn(3, 2).M]
    ipso, otherC, Hs = [], [], []
    for G in gen:
        ipso.append(G @ ringC[0])
        otherC += [G @ p for p in ringC[1:]]
        Hs += [G @ p for p in ringH]
    elems = ['B'] + ['C'] * 18 + ['H'] * 15
    X = [np.zeros(3)] + ipso + otherC + Hs
    axes = [np.array([math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3), 0])
            for k in range(3)]
    zax = 'ось z: перпендикулярна плоскости BC3, проходит через атом B (1)'
    ops = [E().at('элемент не нужен — все атомы остаются на месте'),
           Cn(3, 1).at(zax), Cn(3, 2).at(zax),
           Cn(2, 1, axis=axes[0], sym=r"$C_2'$", desc='поворот на 180° вокруг связи B–C(2)').at(
               'ось x: в плоскости BC3, вдоль связи B(1)–C(2) и через центр кольца 1'),
           Cn(2, 1, axis=axes[1], sym=r"$C_2''$", desc='поворот на 180° вокруг связи B–C(3)').at(
               'в плоскости xy под углом 120° к оси x: вдоль связи B(1)–C(3), через центр кольца 2'),
           Cn(2, 1, axis=axes[2], sym=r"$C_2'''$", desc='поворот на 180° вокруг связи B–C(4)').at(
               'в плоскости xy под углом 240° к оси x: вдоль связи B(1)–C(4), через центр кольца 3')]
    absent = [sigma(Z, r'$\sigma_h$', 'плоскость BC3 (xy)'),
              sigma([0, 1, 0], r'$\sigma_v$', 'плоскость через ось z и B–C1'),
              Sn(3, 1), inv()]
    elements = [('axis', Z, r'$C_3$ (ось z)'),
                ('axis', axes[0], r"$C_2'$ (ось x)"),
                ('axis', axes[1], r"$C_2''$"),
                ('axis', axes[2], r"$C_2'''$")]
    return Molecule('tpb', 'Трифенилборан', 'B(C$_6$H$_5$)$_3$', r'$D_3$',
                    elems, X, ops, absent, elements, labels=[0, 1, 2, 3],
                    size=0.6, view=(30, -70), chiral_view=(85, -90),
                    note=f'«пропеллер», кольца повёрнуты на {TPB_TWIST:.0f}° — хиральная')


TWIST = 40.0


def tetraphenylmethane(twist=TWIST):
    rCC1, rAr, rCH = 1.51, 1.39, 1.08
    d = unit([1, 1, 1])
    e0 = unit(Z - np.dot(Z, d) * d)
    w = math.cos(twist * DEG) * e0 + math.sin(twist * DEG) * np.cross(d, e0)
    ctr = (rCC1 + rAr) * d
    ringC, ringH = [], []
    for m in range(6):
        al = m * math.pi / 3
        r = math.cos(al) * (-d) + math.sin(al) * w
        ringC.append(ctr + rAr * r)
        if m:
            ringH.append(ctr + (rAr + rCH) * r)
    gen = [np.eye(3), Sn(4, 1).M, Cn(2, 1).M, Sn(4, 3).M]
    ipso, otherC, Hs = [], [], []
    for G in gen:
        ipso.append(G @ ringC[0])
        otherC += [G @ p for p in ringC[1:]]
        Hs += [G @ p for p in ringH]
    elems = ['C'] * (1 + 4 + 20) + ['H'] * 20
    X = [np.zeros(3)] + ipso + otherC + Hs
    zax = ('ось z: проходит через центральный атом C (1) и делит пополам углы '
           'между связями C1–C2, C1–C4 (вверх) и C1–C3, C1–C5 (вниз)')
    ops = [E().at('элемент не нужен — все атомы остаются на месте'),
           Sn(4, 1).at(zax + '; плоскость σh — xy (через атом 1)'),
           Cn(2, 1).at(zax),
           Sn(4, 3).at(zax + '; плоскость σh — xy (через атом 1)')]
    absent = [Cn(4, 1),
              sigma([1, -1, 0], r'$\sigma_d$', 'плоскость x = y (через ось z и C–C1)'),
              sigma([0, 1, 0], r'$\sigma$', 'плоскость xz'),
              Cn(2, 1, axis=[1, 0, 0], sym=r"$C_2'$", desc='ось C2 вдоль x'),
              inv()]
    elements = [('axis', Z, r'$S_4,\ C_2$ (ось z)'),
                ('plane', Z, r'$\sigma_h$ — не элемент', 'ghost')]
    return Molecule('tpm', 'Тетрафенилметан', 'C(C$_6$H$_5$)$_4$', r'$S_4$',
                    elems, X, ops, absent, elements, labels=[0, 1, 2, 3, 4],
                    size=0.55, view=(22, 20),
                    note=f'кольца повёрнуты на {TWIST:.0f}°')


def all_molecules():
    return [boric_acid(), triphenylborane(), tetraphenylmethane()]

#проверка

TOL = 1e-3


def match(elems, X, Y, tol=TOL):
    n = len(elems)
    perm, used, dev = [None] * n, set(), 0.0
    for i in range(n):
        best, bj = 1e9, None
        for j in range(n):
            if elems[j] != elems[i]:
                continue
            dist = np.linalg.norm(Y[i] - X[j])
            if dist < best:
                best, bj = dist, j
        dev = max(dev, best)
        if best < tol and bj not in used:
            perm[i] = bj
            used.add(bj)
    if any(p is None for p in perm):
        return None, dev
    return perm, dev


def cycles(perm, only=None):
    seen, out = set(), []
    for s in range(len(perm)):
        if s in seen:
            continue
        c, j = [], s
        while j not in seen:
            seen.add(j)
            c.append(j)
            j = perm[j]
        if only is None or c[0] in only:
            out.append(c)
    return ''.join('(' + ' '.join(str(k + 1) for k in c) + ')' for c in out)


def check_op(mol, op):
    Y = op.apply(mol.X)
    perm, dev = match(mol.elems, mol.X, Y)
    return perm, dev


def plain(s):
    rep = {r'$': '', '_{': '', '^{': '^', '}': '', '_': '', r'\sigma': 'σ',
           r'\ ': ' ', ',': ','}
    for a, b in rep.items():
        s = s.replace(a, b)
    return s


def report_checks(mols=None, stream=sys.stdout):
    mols = mols or all_molecules()
    for mol in mols:
        print(f'\n=== {mol.name} {plain(mol.formula)}: группа {plain(mol.group)}, '
              f'порядок {len(mol.ops)}, атомов {len(mol.elems)} ===', file=stream)
        print('  Операции группы:', file=stream)
        for op in mol.ops:
            perm, dev = check_op(mol, op)
            ok = perm is not None
            p = cycles(perm) if ok else '—'
            if ok and len(mol.elems) > 12:
                moved = sum(1 for i, j in enumerate(perm) if i != j)
                kmoved = sum(1 for i in mol.labels if perm[i] != i)
                p = cycles(perm, only=set(mol.labels)) + f'  (+ ещё {moved - kmoved} атомов)' if moved else '(тождественная)'
            print(f'    {plain(op.sym):6s} {op.desc:42s} '
                  f'{"СОВПАДАЕТ" if ok else "НЕТ"}   откл. {dev:.1e} Å   {p}',
                  file=stream)
        print('  Проверка «лишних» элементов (их быть не должно):', file=stream)
        for op in mol.absent:
            perm, dev = check_op(mol, op)
            print(f'    {plain(op.sym):6s} {op.desc:42s} '
                  f'{"совпадает (!)" if perm else "не совпадает"}   '
                  f'смещение до {dev:.2f} Å', file=stream)
        Ms = [op.M for op in mol.ops]
        closed = all(any(np.allclose(A @ B, C, atol=1e-9) for C in Ms)
                     for A in Ms for B in Ms)
        print(f'  Замкнутость множества операций: {"да" if closed else "НЕТ"}',
              file=stream)


#хиральность

MIRROR_N = np.array([1.0, 0.0, 0.0])
CHIRAL_VIEW = (14, -72)


def axis_angle(R):
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if ang < 1e-9:
        return Z.copy(), 0.0
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, np.argmin(abs(w - 1))])
    ax = unit(ax)
    if not np.allclose(rot(ax, ang), R, atol=1e-6):
        ang = -ang
    return ax, ang


def kabsch(P, T):
    H = P.T @ T
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Q = Vt.T @ np.diag([1, 1, d]) @ U.T
    rmsd = math.sqrt(np.mean(np.sum((P @ Q.T - T) ** 2, axis=1)))
    return Q, rmsd


def _assign(A, B):
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(D)
        return c[np.argsort(r)]
    except ImportError:
        out, used = [], set()
        for i in range(len(A)):
            for j in np.argsort(D[i]):
                if j not in used:
                    used.add(j)
                    out.append(j)
                    break
        return np.array(out)


def best_proper_fit(P, T, elems, iters=40):
    groups = {}
    for i, e in enumerate(elems):
        groups.setdefault(e, []).append(i)
    starts = []
    for k in range(36):
        for flip in (np.eye(3), rot([1, 0, 0], math.pi), rot([0, 1, 0], math.pi / 2),
                     rot([1, 0, 0], math.pi / 2)):
            starts.append(rot(Z, 2 * math.pi * k / 36) @ flip)
    best = (np.eye(3), 1e9)
    for Q in starts:
        prev = None
        for _ in range(iters):
            Y = P @ Q.T
            src, dst = [], []
            for idx in groups.values():
                idx = np.array(idx)
                src += list(idx)
                dst += list(idx[_assign(Y[idx], T[idx])])
            Q, r = kabsch(P[src], T[dst])
            if prev is not None and abs(prev - r) < 1e-10:
                break
            prev = r
        if r < best[1]:
            best = (Q, r)
    return best


def superpose_mirror(mol):
    s = refl(MIRROR_N)
    Xm = mol.X @ s.T
    for op in mol.ops:
        if np.linalg.det(op.M) < 0:
            Q = op.M @ s
            return Q, 0.0, op
    Q, r = best_proper_fit(Xm, mol.X, mol.elems)
    return Q, r, None

#фазы анимации

def ease(s):
    return s * s * (3 - 2 * s)


class Phase:

    def __init__(self, caption, f, dur, element=None):
        self.caption, self.f, self.dur, self.element = caption, f, dur, element


def ph_rotate(axis, angle, caption, dur=1.6, element=None):
    def f(X, s):
        return X @ rot(axis, angle * s).T
    return Phase(caption, f, dur, element)


def ph_reflect(normal, caption, offset=0.0, dur=1.2, element=None):
    n = unit(normal)

    def f(X, s):
        return X - 2 * s * ((X @ n) - offset)[:, None] * n
    return Phase(caption, f, dur, element)


def ph_invert(caption, dur=1.3, element=None):
    return Phase(caption, lambda X, s: X * (1 - 2 * s), dur, element)


def ph_translate(v, caption, dur=1.0):
    v = np.asarray(v, float)
    return Phase(caption, lambda X, s: X + s * v, dur)


def ph_hold(caption, dur=0.8, element=None):
    return Phase(caption, lambda X, s: X, dur, element)


def op_phases(op, mol=None):
    hold_txt = ''
    if op.kind == 'S':
        hold_txt = f'{op.sym}: после поворота — промежуточное положение'
        if mol is not None:
            perm, _ = match(mol.elems, mol.X, mol.X @ rot(op.vec, op.angle).T)
            hold_txt = (f'{op.sym}: после одного поворота модель ещё НЕ совпала — нужно отражение'
                        if perm is None else
                        f'{op.sym}: поворот {op.rot_sym} уже совмещает модель (он есть в группе), '
                        f'отражение σh — тоже')
    if op.kind == 'E':
        return [ph_hold('E: все атомы остаются на месте', 1.0)]
    if op.kind == 'C':
        return [ph_rotate(op.vec, op.angle,
                          f'{op.sym}: {op.desc}', element=('axis', op.vec, op.sym))]
    if op.kind == 's':
        return [ph_reflect(op.vec, f'{op.sym}: {op.desc}',
                           element=('plane', op.vec, op.sym))]
    if op.kind == 'i':
        return [ph_invert(f'{op.sym}: каждый атом → через центр в точку (−x, −y, −z)',
                          element=('point', None, op.sym))]
    if op.kind == 'S':
        deg = round(math.degrees(op.angle))
        return [ph_rotate(op.vec, op.angle,
                          f'{op.sym}, шаг 1: поворот {op.rot_sym} на {deg}°',
                          element=('axis', op.vec, op.sym)),
                ph_hold(hold_txt, 0.8, element=('axis', op.vec, op.sym)),
                ph_reflect(op.vec, f'{op.sym}, шаг 2: отражение в плоскости '
                           r"$\sigma_h$, перпендикулярной оси", element=('plane', op.vec, r'$\sigma_h$'))]
    raise ValueError(op.kind)


#оформление и отрисовка

BG = '#0E1626'
PANEL = '#16213A'
FG = '#E6EAF2'
ACCENT = '#F6C453'
PLANE = '#5FB3F9'
POINT = '#FF7AC6'
MUTED = '#7E8AA6'
AXISC = '#A9B4CC'
LABELC = '#FFD866'
BOND = '#AEB6C6'
OKC = '#9BE59B'
BADC = '#FF8A80'
MIRRORC = '#C792EA'


def light_theme():
    g = globals()
    g.update(BG='#FFFFFF', PANEL='#F3F5F9', FG='#1C2233', AXISC='#4B5468', LABELC='#C2185B', ACCENT='#C27C0E',
             PLANE='#2F80ED', POINT='#D6336C', MUTED='#8A93A8', BOND='#4A5263',
             OKC='#2E8B3D', BADC='#C62828', MIRRORC='#8E44AD')
    COLORS.update(H='#FFFFFF', C='#3B4252')


def perp_basis(n):
    n = unit(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(n, a))
    return u, np.cross(n, u)


class ModelArtist:

    def __init__(self, ax, mol, alpha=1.0, bond_color=None, label_color=None,
                 ghost=False, show_labels=True, zorder=2):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        bond_color = bond_color or BOND
        label_color = label_color or LABELC
        self.ax, self.mol = ax, mol
        X = mol.X
        cols = [COLORS[e] for e in mol.elems]
        sz = [SIZES[e] * mol.size for e in mol.elems]
        if ghost:
            self.sc = ax.scatter(*X.T, s=sz, c=cols, alpha=alpha * 0.9,
                                 edgecolors=MUTED, linewidths=0.6,
                                 depthshade=False)
            self.sc.set_facecolor('none')
        else:
            self.sc = ax.scatter(*X.T, s=sz, c=cols, alpha=alpha,
                                 edgecolors='#0B111C' if BG != '#FFFFFF' else '#3B4252',
                                 linewidths=0.8,
                                 depthshade=False)
        self.lc = Line3DCollection(self._segs(X), colors=bond_color,
                                   linewidths=1.2 if ghost else 2.6,
                                   alpha=alpha * (0.6 if ghost else 0.95),
                                   linestyles='--' if ghost else '-')
        ax.add_collection3d(self.lc)
        self.texts = []
        if show_labels:
            for i in mol.labels:
                off = 0.18 if not ghost else -0.22
                t = ax.text(*(X[i] + off), str(i + 1),
                            color=label_color, fontsize=8 if ghost else 9,
                            fontweight='normal' if ghost else 'bold',
                            ha='center', va='center', zorder=10)
                self.texts.append((i, t, off))

    def _segs(self, X):
        return [[X[i], X[j]] for i, j in self.mol.bonds]

    def update(self, X):
        self.sc._offsets3d = (X[:, 0], X[:, 1], X[:, 2])
        self.lc.set_segments(self._segs(X))
        for i, t, off in self.texts:
            t.set_position_3d(X[i] + off)

    def set_visible(self, v):
        self.sc.set_visible(v)
        self.lc.set_visible(v)
        for _, t, _ in self.texts:
            t.set_visible(v)

    def set_labels_visible(self, v):
        for _, t, _ in self.texts:
            t.set_visible(v)


def draw_element(ax, el, L, color=None, center=np.zeros(3), alpha=0.18,
                 fontsize=12):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    kind, vec, label = el[0], el[1], el[2]
    faded = len(el) > 3 and el[3] == 'ghost'
    arts = []
    if kind == 'axis':
        v = unit(vec)
        c = color or ACCENT
        p1, p2 = center - L * v, center + L * v
        arts += ax.plot(*np.array([p1, p2]).T, color=c, lw=2.2, ls=(0, (6, 4)),
                        alpha=0.95, zorder=1)
        arts.append(ax.text(*(p1 - 0.25 * v), label, color=c, fontsize=fontsize,
                            fontweight='bold', ha='center'))
    elif kind == 'plane':
        n = unit(vec)
        u, w = perp_basis(n)
        c = color or PLANE
        corners = [center + L * (a * u + b * w)
                   for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        poly = Poly3DCollection([corners], facecolors=c,
                                alpha=alpha * (0.45 if faded else 1.0),
                                edgecolors=c, linewidths=1.0,
                                linestyles='--' if faded else '-')
        ax.add_collection3d(poly)
        arts.append(poly)
        arts.append(ax.text(*corners[2], label, color=c, fontsize=fontsize - 1,
                            alpha=0.7 if faded else 1.0))
    elif kind == 'point':
        c = color or POINT
        arts.append(ax.scatter([center[0]], [center[1]], [center[2]], s=160,
                               marker='*', color=c, zorder=20, depthshade=False))
        arts.append(ax.text(*(center + 0.35), label, color=c, fontsize=fontsize))
    return arts


def draw_coord_axes(ax, length, fontsize=12):
    arts = []
    for v, name in zip(np.eye(3), 'xyz'):
        arts += ax.plot(*np.array([-length * v, length * v]).T, color=AXISC, lw=1.0,
                        alpha=0.8, zorder=0)
        arts.append(ax.text(*(length * 1.06 * v), name, color=AXISC, fontsize=fontsize,
                            fontweight='bold', ha='center', va='center',
                            fontstyle='italic'))
    return arts


def style_3d(ax, L, center=np.zeros(3), axes_triad=False, zoom=1.35, coord_axes=True,
             axes_len=None):
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a.line.set_color((0, 0, 0, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlim(center[0] - L, center[0] + L)
    ax.set_ylim(center[1] - L, center[1] + L)
    ax.set_zlim(center[2] - L, center[2] + L)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    arts = []
    if coord_axes:
        arts += draw_coord_axes(ax, axes_len or 0.88 * L)
    if axes_triad:
        o = center + np.array([-L, -L, -L]) * 0.92
        for v, name in zip(np.eye(3), 'xyz'):
            arts += ax.plot(*np.array([o, o + 0.28 * L * v]).T, color=MUTED, lw=1.2)
            arts.append(ax.text(*(o + 0.34 * L * v), name, color=MUTED, fontsize=9))
    return arts


#приложение


class App:
    def __init__(self, fig=None, start=0):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, Button, CheckButtons
        self.plt = plt
        self.mols = all_molecules()
        self.fig = fig or plt.figure(figsize=(14, 8.4))
        self.fig.patch.set_facecolor(BG)
        try:
            self.fig.canvas.manager.set_window_title('Симметрия молекул — РГР №1')
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.25, 0.02, 0.74, 0.90], projection='3d')
        self.title = self.fig.text(0.62, 0.955, '', color=FG, fontsize=15,
                                   ha='center', va='center', fontweight='bold')
        self.caption = self.fig.text(0.62, 0.915, '', color=ACCENT, fontsize=12,
                                     ha='center', va='center')
        self.subcap = self.fig.text(0.62, 0.878, '', color=FG, fontsize=10,
                                    ha='center', va='center', alpha=0.85)

        def panel(rect, title):
            a = self.fig.add_axes(rect, facecolor=PANEL)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color('#2A3656')
            self.fig.text(rect[0] + 0.005, rect[1] + rect[3] + 0.008, title,
                          color=MUTED, fontsize=9)
            return a

        self.ax_mol = panel([0.012, 0.74, 0.22, 0.20], 'МОЛЕКУЛА')
        labels = [f'{m.name} — {m.group}' for m in self.mols]
        self.r_mol = RadioButtons(self.ax_mol, labels, active=start,
                                  activecolor=ACCENT)
        self._style_radio(self.r_mol, 10)
        self.r_mol.on_clicked(self.on_molecule)

        self.ops_rect = [0.012, 0.335, 0.22, 0.37]
        self.ax_ops_title = self.fig.text(self.ops_rect[0] + 0.005,
                                          self.ops_rect[1] + self.ops_rect[3] + 0.008,
                                          'ОПЕРАЦИЯ', color=MUTED, fontsize=9)
        self.ax_ops = None
        self.r_ops = None

        def button(rect, text, cb, color='#24345A'):
            a = self.fig.add_axes(rect)
            b = Button(a, text, color=color, hovercolor='#33497A')
            b.label.set_color(FG)
            b.label.set_fontsize(10)
            b.on_clicked(cb)
            return b

        self.b_apply = button([0.012, 0.268, 0.107, 0.05], '▶  Применить', self.on_apply,
                              color='#2D5A3D')
        self.b_back = button([0.125, 0.268, 0.107, 0.05], '↺  Вернуть', self.on_back)
        self.b_check = button([0.012, 0.21, 0.107, 0.05], '✓  Проверить всё',
                              self.on_check_all)
        self.b_chiral = button([0.125, 0.21, 0.107, 0.05], '⇋  Зеркало',
                               self.on_chiral, color='#4A2D5A')
        self.ax_cb = self.fig.add_axes([0.012, 0.150, 0.22, 0.05], facecolor=PANEL)
        self.cb = CheckButtons(self.ax_cb, ['исходная модель', 'номера атомов'],
                               [True, True])
        for t in self.cb.labels:
            t.set_color(FG)
            t.set_fontsize(9)
        for s in self.ax_cb.spines.values():
            s.set_color('#2A3656')
        self.cb.on_clicked(self.on_toggle)

        self.ax_info = panel([0.012, 0.015, 0.22, 0.10], 'РЕЗУЛЬТАТ ПРОВЕРКИ')
        self.info = self.ax_info.text(0.03, 0.93, '', color=FG, fontsize=8.5,
                                      va='top', ha='left', wrap=True,
                                      transform=self.ax_info.transAxes,
                                      family='DejaVu Sans')

        self.timer = self.fig.canvas.new_timer(interval=25)
        self.timer.add_callback(self._tick)
        self.anim = None
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.set_molecule(start)

    def _style_radio(self, r, fs):
        for t in r.labels:
            t.set_color(FG)
            t.set_fontsize(fs)
        for s in r.ax.spines.values():
            s.set_color('#2A3656')

    def on_molecule(self, label):
        idx = [f'{m.name} — {m.group}' for m in self.mols].index(label)
        self.set_molecule(idx)

    def set_molecule(self, idx):
        from matplotlib.widgets import RadioButtons
        self.stop()
        self.mol = self.mols[idx]
        self.mode = 'ops'
        self.op_idx = 0
        if self.ax_ops is not None:
            self.ax_ops.remove()
        self.ax_ops = self.fig.add_axes(self.ops_rect, facecolor=PANEL)
        self.ax_ops.set_xticks([])
        self.ax_ops.set_yticks([])
        labels = [f'{op.sym}  {op.desc}' for op in self.mol.ops]
        self.r_ops = RadioButtons(self.ax_ops, labels, active=0, activecolor=ACCENT)
        self._style_radio(self.r_ops, 9.5)
        self.r_ops.on_clicked(self.on_op)
        self.ax_ops_title.set_text(f'ОПЕРАЦИЯ  (группа {self.mol.group}, '
                                   f'порядок {len(self.mol.ops)})')
        self.redraw_scene()
        self.set_info(f'{self.mol.name}: {self.mol.note}.\n'
                      f'Выберите операцию и нажмите «Применить» (пробел).\n'
                      f'Мышью можно вращать сцену.')
        self.fig.canvas.draw_idle()

    def redraw_scene(self, center=np.zeros(3), L=None):
        self.ax.cla()
        self.L = L or (self.mol.extent + 0.9)
        style_3d(self.ax, self.L, center,
                 axes_len=(self.mol.extent + 0.5) if np.any(center) else None)
        self.ax.view_init(*self.mol.view)
        self.ghost = ModelArtist(self.ax, self.mol, alpha=0.55, ghost=True,
                                 label_color=MUTED)
        self.main = ModelArtist(self.ax, self.mol)
        self.image = None
        self.el_arts = []
        self.Xcur = self.mol.X.copy()
        self.transformed = False
        self.title.set_text(f'{self.mol.name}  {self.mol.formula}   ·   '
                            f'точечная группа {self.mol.group}')
        self.caption.set_text('')
        self.subcap.set_text('')
        self.on_toggle(None)

    def on_op(self, label):
        labels = [f'{op.sym}  {op.desc}' for op in self.mol.ops]
        self.op_idx = labels.index(label)
        if not self.anim:
            self.snap_back()
            self.show_element(op_phases(self.mol.ops[self.op_idx])[0].element)
            op = self.mol.ops[self.op_idx]
            self.caption.set_text(f'выбрана {op.sym}: {op.desc}')
            self.show_where(op)
            self.fig.canvas.draw_idle()

    def show_where(self, op):
        self.subcap.set_text(f'{op.where}\n{op.formula}' if op.kind != 'E' else op.where)

    def show_element(self, el, center=np.zeros(3)):
        for a in self.el_arts:
            try:
                a.remove()
            except Exception:
                pass
        self.el_arts = []
        if el is not None:
            self.el_arts = draw_element(self.ax, el, self.L * 0.85, center=center)

    def play(self, phases, X0, on_done=None, reverse=False):
        B = [X0]
        for p in phases:
            B.append(p.f(B[-1], 1.0))
        seq = list(range(len(phases)))
        if reverse:
            seq = seq[::-1]
        self.anim = dict(phases=phases, B=B, seq=seq, pos=0, t=0.0,
                         reverse=reverse, done=on_done)
        self._enter_phase()
        self.timer.start()

    def _enter_phase(self):
        a = self.anim
        p = a['phases'][a['seq'][a['pos']]]
        self.show_element(p.element, center=getattr(self, 'el_center', np.zeros(3)))
        self.caption.set_text(('↺ ' if a['reverse'] else '') + p.caption)

    def _tick(self):
        a = self.anim
        if a is None:
            return
        k = a['seq'][a['pos']]
        p = a['phases'][k]
        a['t'] += 0.025 / p.dur
        s = ease(min(a['t'], 1.0))
        if a['reverse']:
            s = 1 - s
        X = p.f(a['B'][k], s)
        self._set_X(X)
        if a['t'] >= 1.0:
            a['pos'] += 1
            a['t'] = 0.0
            if a['pos'] >= len(a['seq']):
                self.timer.stop()
                done = a['done']
                self.anim = None
                if done:
                    done()
            else:
                self._enter_phase()
        self.fig.canvas.draw_idle()

    def _set_X(self, X):
        self.Xcur = X
        target = self.image if self.mode == 'chiral' else self.main
        target.update(X)

    def stop(self):
        try:
            self.timer.stop()
        except Exception:
            pass
        self.anim = None

    def snap_back(self):
        if self.mode == 'chiral':
            self.mode = 'ops'
            self.redraw_scene()
        self.Xcur = self.mol.X.copy()
        self.main.update(self.Xcur)
        self.transformed = False

    def on_apply(self, _=None):
        if self.anim:
            return
        self.snap_back()
        op = self.mol.ops[self.op_idx]
        self.cur_phases = op_phases(op, self.mol)
        self.show_where(op)
        self.play(self.cur_phases, self.mol.X, on_done=lambda: self._after(op))

    def _after(self, op):
        self.transformed = True
        perm, dev = match(self.mol.elems, self.mol.X, self.Xcur)
        if perm is None:
            self.set_info(f'{op.sym}: модель НЕ совпала с исходной (откл. {dev:.2f} Å)')
            return
        moved = sum(1 for i, j in enumerate(perm) if i != j)
        cyc = cycles(perm, only=set(self.mol.labels) if len(perm) > 12 else None)
        self.set_info(f'{op.sym}: ✓ модель совпала с исходной\n'
                      f'откл. {dev:.1e} Å, смещено атомов: {moved}\n'
                      f'перестановка: {cyc}')
        self.caption.set_text(f'{op.sym}: результат совпадает с исходной моделью')

    def on_back(self, _=None):
        if self.anim:
            return
        if self.mode == 'chiral':
            self.snap_back()
            self.set_info('Возврат к исходной модели.')
            self.fig.canvas.draw_idle()
            return
        if not self.transformed:
            return
        self.play(self.cur_phases, self.mol.X,
                  on_done=lambda: (setattr(self, 'transformed', False),
                                   self.caption.set_text('возврат в исходное состояние'),
                                   self.set_info('Модель возвращена в исходное состояние.')),
                  reverse=True)

    def on_check_all(self, _=None):
        if self.anim:
            return
        ok = []
        for op in self.mol.ops:
            perm, dev = check_op(self.mol, op)
            ok.append(perm is not None)
        bad = []
        for op in self.mol.absent:
            perm, dev = check_op(self.mol, op)
            bad.append(perm is None)
        line1 = ' '.join(f'{op.sym}✓' if g else f'{op.sym}✗'
                         for op, g in zip(self.mol.ops, ok))
        line2 = ' '.join(f'{op.sym}' for op in self.mol.absent)
        self.set_info(f'Все {sum(ok)} из {len(ok)} операций совмещают модель:\n{line1}\n'
                      f'Не совмещают (нет в группе): {line2}')
        report_checks([self.mol])
        self.fig.canvas.draw_idle()

    def on_chiral(self, _=None):
        if self.anim:
            return
        mol = self.mol
        self.mode = 'chiral'
        d = mol.extent + 0.8
        center = np.array([d, 0, 0])
        self.redraw_scene(center=center, L=d + mol.extent + 0.3)
        self.ax.view_init(*(mol.chiral_view or CHIRAL_VIEW))
        self.mode = 'chiral'
        self.ghost.set_visible(False)
        self.image = ModelArtist(self.ax, mol, bond_color=MIRRORC)
        # зеркало
        draw_element(self.ax, ('plane', MIRROR_N, 'зеркало'), mol.extent + 0.9,
                     color=MIRRORC, center=np.array([d, 0, 0]), alpha=0.12)
        Q, rmsd, sop = superpose_mirror(mol)
        ax_, ang = axis_angle(Q)
        phases = [
            ph_reflect(MIRROR_N, 'шаг 1: отражение в зеркале — получаем зеркальное изображение',
                       offset=d, dur=1.6),
            ph_hold('шаг 2: перенесём изображение на место исходной молекулы', 0.4),
            ph_translate(-2 * d * MIRROR_N, 'шаг 2: перенос изображения к исходной молекуле',
                         dur=1.2),
            ph_rotate(ax_, ang, f'шаг 3: пробуем совместить поворотом на '
                      f'{abs(round(math.degrees(ang)))}°', dur=1.8,
                      element=('axis', ax_, 'поворот')),
        ]

        def done():
            perm, dev = match(mol.elems, mol.X, self.Xcur, tol=1e-2)
            if perm is not None:
                why = (f'(благодаря элементу {sop.sym}: Q = {sop.sym}·σ)'
                       if sop is not None else '')
                self.caption.set_text('изображение СОВМЕСТИЛОСЬ с молекулой → ахиральна')
                self.set_info(f'Зеркальное изображение совмещается\nс исходной поворотом '
                              f'{why}.\nМолекула АХИРАЛЬНА.')
            else:
                self.caption.set_text('изображение НЕ совмещается ни при каком повороте '
                                      '→ хиральна')
                self.set_info(f'Лучший поворот оставляет\nсреднеквадр. отклонение '
                              f'{rmsd:.2f} Å.\nНесобственных операций нет — '
                              f'молекула ХИРАЛЬНА.')
        self.play(phases, mol.X, on_done=done)

    def on_toggle(self, _):
        st = self.cb.get_status()
        if self.mode != 'chiral':
            self.ghost.set_visible(st[0])
            self.ghost.set_labels_visible(st[0] and st[1])
        self.main.set_labels_visible(st[1])
        if self.image is not None:
            self.image.set_labels_visible(st[1])
        self.fig.canvas.draw_idle()

    def on_key(self, ev):
        if ev.key == ' ':
            self.on_apply()
        elif ev.key in ('r', 'к'):
            self.on_back()
        elif ev.key in ('up', 'down') and self.r_ops is not None:
            n = len(self.mol.ops)
            i = (self.op_idx + (1 if ev.key == 'down' else -1)) % n
            self.r_ops.set_active(i)

    def set_info(self, s):
        self.info.set_text(s)


def draw_arc(ax, axis, angle, L, color=None):
    if abs(angle) < 1e-6:
        return
    n = unit(axis)
    u, _ = perp_basis(n)
    w = np.cross(n, u)
    r, h = 0.32 * L, 0.80 * L
    t = np.linspace(0, angle, 60)
    P = h * n + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)
    c = color or ACCENT
    ax.plot(*P.T, color=c, lw=2.2, zorder=30)
    d = P[-1] - P[-4]
    tip = P[-1] + 0.6 * d / np.linalg.norm(d) * 0.18 * L
    ax.plot(*np.array([P[-1], tip]).T, color=c, lw=2.2)
    ax.scatter(*tip, s=40, marker='o', color=c, depthshade=False)
    ax.text(*(h * n + r * 1.15 * (math.cos(angle / 2) * u + math.sin(angle / 2) * w)),
            f'{abs(round(math.degrees(angle)))}°', color=c, fontsize=10, fontweight='bold')


def strip_frames(mol, op):
    X0 = mol.X
    fr = []
    if op.kind == 'E':
        return [('исходная = результат', X0, [], 0, None)]
    if op.kind == 'C':
        el = ('axis', op.vec, '')
        for f in (0, 1 / 3, 2 / 3, 1):
            a = op.angle * f
            X = X0 @ rot(op.vec, a).T
            t = 'исходная' if f == 0 else f'поворот на {round(math.degrees(a))}°'
            fr.append((t, X, [el], a, op.vec))
        return fr
    if op.kind == 's':
        el = ('plane', op.vec, '')
        n = op.vec
        if np.allclose(X0 @ n, 0, atol=1e-6):
            return [('исходная', X0, [el], 0, None),
                    ('все атомы в плоскости → на месте', X0, [el], 0, None)]
        for f, t in ((0, 'исходная'), (0.5, 'середина: атомы в плоскости'), (1, 'после отражения')):
            fr.append((t, X0 - 2 * f * (X0 @ n)[:, None] * n, [el], 0, None))
        return fr
    if op.kind == 'i':
        el = ('point', None, '')
        for f, t in ((0, 'исходная'), (0.5, 'середина: все атомы в центре'), (1, 'после инверсии')):
            fr.append((t, X0 * (1 - 2 * f), [el], 0, None))
        return fr
    if op.kind == 'S':
        ax_el, pl = ('axis', op.vec, ''), ('plane', op.vec, '')
        deg = round(math.degrees(op.angle))
        X1 = X0 @ rot(op.vec, op.angle).T
        n = op.vec
        fr.append(('исходная', X0, [ax_el], 0, None))
        fr.append((f'шаг 1: поворот на {deg // 2}°', X0 @ rot(op.vec, op.angle / 2).T, [ax_el],
                   op.angle / 2, op.vec))
        fr.append((f'шаг 1: поворот на {deg}°', X1, [ax_el], op.angle, op.vec))
        fr.append(('шаг 2: середина отражения σh', X1 - (X1 @ n)[:, None] * n, [ax_el, pl], 0, None))
        fr.append(('шаг 2: после σh', X1 - 2 * (X1 @ n)[:, None] * n, [ax_el, pl], 0, None))
        return fr
    raise ValueError(op.kind)


def export_strips(outdir):
    import matplotlib.pyplot as plt
    saved = []
    for mol in all_molecules():
        for k, op in enumerate(mol.ops):
            frames = strip_frames(mol, op)
            n = len(frames)
            fig = plt.figure(figsize=(max(3.05 * n, 7.6), 3.2), facecolor=BG)
            L = mol.extent + 0.8
            for j, (title, X, els, arc, arcax) in enumerate(frames):
                ax = fig.add_subplot(1, n, j + 1, projection='3d')
                style_3d(ax, L, zoom=1.55)
                ax.view_init(*mol.view)
                for el in els:
                    draw_element(ax, el, L * 0.85, alpha=0.14)
                if arcax is not None:
                    draw_arc(ax, arcax, arc, L)
                if j:
                    ModelArtist(ax, mol, ghost=True, alpha=0.5, label_color=MUTED)
                art = ModelArtist(ax, mol)
                art.update(X)
                last = j == n - 1
                ok = ''
                if last and op.kind != 'E':
                    ok = '  ✓ совпала' if match(mol.elems, mol.X, X)[0] else '  ✗ не совпала'
                if op.kind == 'S' and j == 2:
                    ok = '  ✓' if match(mol.elems, mol.X, X)[0] else '  ✗ ещё не совпала'
                ax.set_title(title + ok, color=(BADC if '✗' in ok else FG), fontsize=10.5, pad=-2)
            fig.suptitle(f'{mol.name}: {op.sym} — {op.desc}', color=ACCENT, fontsize=13, y=0.985)
            fig.subplots_adjust(left=0, right=1, bottom=-0.04, top=0.87, wspace=0)
            pth = os.path.join(outdir, f'strip_{mol.key}_{k}.png')
            fig.savefig(pth, dpi=130, facecolor=BG)
            plt.close(fig)
            saved.append(pth)
    return saved


 #экспорт картинок в отчет

def export_figures(outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    mols = all_molecules()
    plt.rcParams['font.family'] = 'DejaVu Sans'

    def new3d(fig, rect_or_pos, mol, L=None, center=np.zeros(3)):
        ax = fig.add_subplot(*rect_or_pos, projection='3d') \
            if isinstance(rect_or_pos, tuple) else fig.add_axes(rect_or_pos,
                                                                projection='3d')
        L = L or mol.extent + 0.8
        style_3d(ax, L, center, axes_len=(mol.extent + 0.5) if np.any(center) else None)
        ax.view_init(*mol.view)
        return ax, L

    saved = []
    #1 модель + элементы симметрии
    for mol in mols:
        fig = plt.figure(figsize=(6.4, 6.0), facecolor=BG)
        ax, L = new3d(fig, [0, 0, 1, 0.93], mol)
        for el in mol.elements:
            draw_element(ax, el, L * 0.85, fontsize=13)
        ModelArtist(ax, mol)
        fig.text(0.5, 0.955, f'{mol.name} {mol.formula} — {mol.group}', color=FG,
                 fontsize=15, ha='center', fontweight='bold')
        p = os.path.join(outdir, f'{mol.key}_elements.png')
        fig.savefig(p, dpi=170, facecolor=BG)
        plt.close(fig)
        saved.append(p)

    #2 сетка: результат каждой операции
    for mol in mols:
        n = len(mol.ops)
        cols = 3 if n == 6 else (4 if n > 4 else n)
        rows = math.ceil(n / cols)
        fig = plt.figure(figsize=(3.4 * cols, 3.5 * rows), facecolor=BG)
        for k, op in enumerate(mol.ops):
            ax, L = new3d(fig, (rows, cols, k + 1), mol)
            el = op_phases(op)[0].element
            if op.kind == 'S':
                draw_element(ax, ('plane', op.vec, ''), L * 0.8, fontsize=9, alpha=0.1)
            if el:
                draw_element(ax, (el[0], el[1], ''), L * 0.85, fontsize=9)
            ModelArtist(ax, mol, ghost=True, alpha=0.55, label_color=MUTED)
            Y = op.apply(mol.X)
            art = ModelArtist(ax, mol)
            art.update(Y)
            perm, dev = match(mol.elems, mol.X, Y)
            ok = '✓ совпадает' if perm is not None else '✗'
            ax.set_title(f'{op.sym}   {ok}', color=FG, fontsize=12, pad=-4)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0, hspace=0.05)
        p = os.path.join(outdir, f'{mol.key}_ops.png')
        fig.savefig(p, dpi=140, facecolor=BG)
        plt.close(fig)
        saved.append(p)

    #3 пошаговая Sn: исходная → после Cn → после σh
    for mol in [m for m in mols if any(o.kind == 'S' for o in m.ops)]:
        op = [o for o in mol.ops if o.kind == 'S'][0]
        X1 = mol.X @ rot(op.vec, op.angle).T
        X2 = op.apply(mol.X)
        fig = plt.figure(figsize=(12, 4.3), facecolor=BG)
        titles = ['исходная модель',
                  f'шаг 1: после поворота {op.rot_sym}',
                  r'шаг 2: после отражения $\sigma_h$']
        for k, (Xk, tt) in enumerate(zip([mol.X, X1, X2], titles)):
            ax, L = new3d(fig, (1, 3, k + 1), mol)
            if k == 1:
                draw_element(ax, ('axis', op.vec, ''), L * 0.85)
            if k == 2:
                draw_element(ax, ('plane', op.vec, ''), L * 0.8, alpha=0.15)
            if k:
                ModelArtist(ax, mol, ghost=True, alpha=0.55, label_color=MUTED)
            art = ModelArtist(ax, mol)
            art.update(Xk)
            perm, _ = match(mol.elems, mol.X, Xk)
            mark = '' if k == 0 else ('  ✓ совпала' if perm else '  ✗ не совпала')
            ax.set_title(tt + mark, color=FG if (k == 0 or perm) else BADC,
                         fontsize=12, pad=-2)
        fig.suptitle(f'{mol.name}: операция {op.sym} = '
                     r'$\sigma_h \cdot$' + f'{op.rot_sym}', color=ACCENT, fontsize=14)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.86, wspace=0)
        p = os.path.join(outdir, f'{mol.key}_S_steps.png')
        fig.savefig(p, dpi=150, facecolor=BG)
        plt.close(fig)
        saved.append(p)

    #4 хиральность: молекула, зеркало, изображение и результат совмещения
    for mol in [mols[2], mols[1]]:
        fig = plt.figure(figsize=(12, 5), facecolor=BG)
        d = mol.extent + 0.8
        Xm = mol.X - 2 * ((mol.X @ MIRROR_N) - d)[:, None] * MIRROR_N
        ax, L = new3d(fig, [0, 0, 0.6, 0.86], mol, L=d + mol.extent + 0.2,
                      center=np.array([d, 0, 0]))
        ax.view_init(*(mol.chiral_view or CHIRAL_VIEW))
        draw_element(ax, ('plane', MIRROR_N, 'зеркало'), mol.extent + 0.9,
                     color=MIRRORC, center=np.array([d, 0, 0]), alpha=0.12)
        ModelArtist(ax, mol)
        im = ModelArtist(ax, mol, bond_color=MIRRORC)
        im.update(Xm)
        ax.set_title('молекула  |  зеркало  |  зеркальное изображение', color=FG,
                     fontsize=12)
        Q, rmsd, sop = superpose_mirror(mol)
        Xs = (Xm - 2 * d * MIRROR_N) @ Q.T
        ax2, L2 = new3d(fig, [0.6, 0, 0.4, 0.86], mol)
        ModelArtist(ax2, mol)
        im2 = ModelArtist(ax2, mol, bond_color=MIRRORC)
        im2.update(Xs)
        perm, _ = match(mol.elems, mol.X, Xs, tol=1e-2)
        res = ('совместились → АХИРАЛЬНА' if perm else
               f'не совмещаются (RMSD {rmsd:.2f} Å) → ХИРАЛЬНА')
        ax2.set_title(f'после переноса и поворота:\n{res}',
                      color=OKC if perm else BADC, fontsize=12)
        fig.suptitle(f'{mol.name} {mol.formula} ({mol.group})', color=ACCENT,
                     fontsize=15, y=0.99)
        p = os.path.join(outdir, f'{mol.key}_chirality.png')
        fig.savefig(p, dpi=150, facecolor=BG)
        plt.close(fig)
        saved.append(p)

    #5 скрин окна программы (середина анимации S4 у тетрафенилметана)
    fig = plt.figure(figsize=(14, 8.4))
    app = App(fig, start=2)
    app.r_ops.set_active(1)
    op = app.mol.ops[1]
    ph = op_phases(op)
    X1 = ph[0].f(app.mol.X, 1.0)
    X = ph[2].f(X1, 0.45)
    app.show_element(ph[2].element)
    app.caption.set_text(ph[2].caption)
    app.main.update(X)
    app.set_info('Пример: середина шага 2 операции S4 —\nатомы движутся сквозь плоскость σh.')
    p = os.path.join(outdir, 'gui_screenshot.png')
    fig.savefig(p, dpi=120, facecolor=BG)
    plt.close(fig)
    saved.append(p)
    saved += export_strips(outdir)
    return saved


def main():
    ap = argparse.ArgumentParser(description='Визуализация симметрии молекул')
    ap.add_argument('--check', action='store_true',
                    help='проверить все операции и вывести таблицу в консоль')
    ap.add_argument('--export', metavar='DIR',
                    help='сохранить рисунки для отчёта в папку DIR')
    ap.add_argument('--light', action='store_true',
                    help='светлая тема рисунков (для печати), вместе с --export')
    ap.add_argument('--molecule', type=int, default=0,
                    help='номер молекулы при запуске (0–2)')
    a = ap.parse_args()
    if a.check:
        report_checks()
        return
    if a.export:
        if a.light:
            light_theme()
        for p in export_figures(a.export):
            print('сохранено:', p)
        return
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['toolbar'] = 'None'
    App(start=a.molecule)
    plt.show()


if __name__ == '__main__':
    main()
