import hashlib
import itertools
import json
import os

import numpy as np
from scipy.optimize import linprog

ATOMS3 = list(itertools.product([1, 0], repeat=3))


def _marginal_row(question_idx, val):
    return np.array([1.0 if atom[question_idx] == val else 0.0 for atom in ATOMS3])


def _solve_min_violation(A, bvec, n_vars):
    m = A.shape[0]
    c_obj = np.concatenate([np.zeros(n_vars), np.ones(m)])
    A_ub = np.zeros((2 * m, n_vars + m))
    b_ub = np.zeros(2 * m)
    for j in range(m):
        A_ub[j, :n_vars] = A[j]
        A_ub[j, n_vars + j] = -1.0
        b_ub[j] = bvec[j]
        A_ub[m + j, :n_vars] = -A[j]
        A_ub[m + j, n_vars + j] = -1.0
        b_ub[m + j] = -bvec[j]
    A_eq = np.concatenate([np.ones((1, n_vars)), np.zeros((1, m))], axis=1)
    return linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=[1.0],
        bounds=[(0, 1)] * n_vars + [(0, None)] * m,
    )


def global_section_lp(marg):
    cons, labels = [], []
    for key, target in sorted(marg.items()):
        qi = "ABC".index(key[-1])
        cons.append((_marginal_row(qi, 1), float(target)))
        labels.append(key)
    A = np.array([c[0] for c in cons])
    bvec = np.array([c[1] for c in cons])
    n = len(ATOMS3)

    res = linprog(
        c=np.zeros(n),
        A_eq=np.vstack([A, np.ones((1, n))]),
        b_eq=np.concatenate([bvec, [1.0]]),
        bounds=[(0, 1)] * n,
    )
    if res.success:
        return dict(feasible=True, eps_star=0.0, certificate=None, n_constraints=len(cons), labels=labels)

    res2 = _solve_min_violation(A, bvec, n)
    eps_star = float(res2.fun) if res2.success else float("nan")
    cert = None
    if res2.success:
        cert = dict(
            ub_dual=np.asarray(res2.ineqlin.marginals, dtype=float).tolist(),
            eq_dual=float(res2.eqlin.marginals[0]),
        )
    return dict(feasible=False, eps_star=eps_star, certificate=cert, n_constraints=len(cons), labels=labels)


def hash_certificate(cert):
    if cert is None:
        return ""
    return hashlib.sha256(json.dumps(cert, sort_keys=True).encode()).hexdigest()[:16]


def cbd_lite_lp(marg, contexts=("ABC", "BCA", "CAB")):
    ks = "ABC"
    copy_list = [(k, c) for k in ks for c in contexts]
    cid = {cp: i for i, cp in enumerate(copy_list)}
    ncopy = len(copy_list)
    natoms = 2 ** ncopy

    def atom_bit(atom_idx, copy_i):
        return (atom_idx >> (ncopy - 1 - copy_i)) & 1

    def marginal_row(cp):
        i = cid[cp]
        return np.array([1.0 if atom_bit(a, i) else 0.0 for a in range(natoms)])

    def disagree_row(cp1, cp2):
        i1, i2 = cid[cp1], cid[cp2]
        return np.array([1.0 if atom_bit(a, i1) != atom_bit(a, i2) else 0.0 for a in range(natoms)])

    rows, targets = [], []
    for k in ks:
        for c in contexts:
            rows.append(marginal_row((k, c)))
            targets.append(float(marg[c + k]))
    for k in ks:
        for c1, c2 in itertools.combinations(contexts, 2):
            rows.append(disagree_row((k, c1), (k, c2)))
            targets.append(abs(float(marg[c1 + k]) - float(marg[c2 + k])))
    A = np.array(rows)
    bvec = np.array(targets)

    res = linprog(
        c=np.zeros(natoms),
        A_eq=np.vstack([A, np.ones((1, natoms))]),
        b_eq=np.concatenate([bvec, [1.0]]),
        bounds=[(0, 1)] * natoms,
    )
    if res.success:
        return dict(feasible=True, eps_star=0.0, n_atoms=natoms)

    res2 = _solve_min_violation(A, bvec, natoms)
    eps = float(res2.fun) if res2.success else float("nan")
    return dict(feasible=False, eps_star=eps, n_atoms=natoms)


def cbd2_lp(J_AB, J_BA):
    pA_AB = J_AB[0] + J_AB[1]
    pB_AB = J_AB[0] + J_AB[2]
    pB_BA = J_BA[0] + J_BA[1]
    pA_BA = J_BA[0] + J_BA[2]
    atoms = list(itertools.product([1, 0], repeat=4))

    def row(pred):
        return np.array([1.0 if pred(t) else 0.0 for t in atoms])

    rows, tgts = [], []
    for (av, bv), cell in zip([(1, 1), (1, 0), (0, 1), (0, 0)], J_AB):
        rows.append(row(lambda t, av=av, bv=bv: t[0] == av and t[1] == bv))
        tgts.append(float(cell))
    for (bv, av), cell in zip([(1, 1), (1, 0), (0, 1), (0, 0)], J_BA):
        rows.append(row(lambda t, bv=bv, av=av: t[2] == bv and t[3] == av))
        tgts.append(float(cell))
    rows.append(row(lambda t: t[0] != t[3]))
    tgts.append(abs(pA_AB - pA_BA))
    rows.append(row(lambda t: t[1] != t[2]))
    tgts.append(abs(pB_AB - pB_BA))

    A = np.array(rows)
    b = np.array(tgts)
    res = linprog(
        np.zeros(len(atoms)),
        A_eq=np.vstack([A, np.ones((1, len(atoms)))]),
        b_eq=np.concatenate([b, [1.0]]),
        bounds=[(0, 1)] * len(atoms),
    )
    return dict(feasible=bool(res.success))


def joints_proper(J_AB, J_BA, tol=1e-9):
    return all(-tol <= c <= 1.0 + tol for c in list(J_AB) + list(J_BA))


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, row):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_ledger(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
