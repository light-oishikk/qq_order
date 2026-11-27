import itertools
import math
import sys
import numpy as np

import analysis as A
import substrates as S
from quantum_core import (
    bloch_to_rho,
    dephase_qubit,
    embed,
    luders_qubit_analytics,
    luders_qubit_table,
    partial_trace_qubit0,
    projector_from_axis,
    random_bloch_axis,
    random_bloch_state,
    random_pure_rho,
    selective_partial_collapse,
    yes_prob,
)

TOL_STRICT = 1e-12
TOL_LOOSE = 1e-10
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())


def check_luders(n=1000, seed=101):
    dmax_j = 0.0
    for k in range(300):
        rng = np.random.default_rng(int.from_bytes(np.random.default_rng(k).bytes(4), "little"))
        r = random_bloch_state(rng)
        a = random_bloch_axis(rng)
        b = random_bloch_axis(rng)
        rho = bloch_to_rho(r)
        Pa = projector_from_axis(a)
        Pb = projector_from_axis(b)
        Qa, Qb = np.eye(2) - Pa, np.eye(2) - Pb
        jAB_dm = [
            float(np.real(np.trace(Pb @ Pa @ rho @ Pa))),
            float(np.real(np.trace(Qb @ Pa @ rho @ Pa))),
            float(np.real(np.trace(Pb @ Qa @ rho @ Qa))),
            float(np.real(np.trace(Qb @ Qa @ rho @ Qa))),
        ]
        jAB, _ = S.luders_qubit_joints(r, a, b)
        dmax_j = max(dmax_j, max(abs(x - y) for x, y in zip(jAB, jAB_dm)))
    record("luders_joints_vs_dm", dmax_j < TOL_STRICT, f"maxdiff={dmax_j:.2e}")

    rng = np.random.default_rng(606)
    diag_spread = 0.0
    for _ in range(200):
        jAB, jBA = S.luders_qubit_joints(
            random_bloch_state(rng), random_bloch_axis(rng), random_bloch_axis(rng)
        )
        diag_spread = max(diag_spread, abs((jAB[0] + jAB[3]) - (jBA[0] + jBA[3])))
    record("luders_diagonal_equality", diag_spread < TOL_STRICT, f"maxdiff={diag_spread:.2e}")

    rng = np.random.default_rng(seed)
    d_table = d_R = d_S = d_dm = 0.0
    for _ in range(n):
        r = random_bloch_state(rng)
        a = random_bloch_axis(rng)
        b = random_bloch_axis(rng)
        t = luders_qubit_table(r, a, b)
        res = S.qq_residuals(t)
        an = luders_qubit_analytics(r, a, b)
        d_table = max(d_table, abs(res["q_A"] - an["q_A"]), abs(res["q_B"] - an["q_B"]))
        d_R = max(d_R, abs(res["R"] - an["R"]))
        d_S = max(d_S, abs(res["S"] - an["S"]))
        rho = bloch_to_rho(r)
        Pa, Pb = projector_from_axis(a), projector_from_axis(b)
        pA2_dm = yes_prob(dephase_qubit(rho, 1, 0, b, 1.0), Pa)
        pB2_dm = yes_prob(dephase_qubit(rho, 1, 0, a, 1.0), Pb)
        d_dm = max(d_dm, abs(t["pA2"] - pA2_dm), abs(t["pB2"] - pB2_dm))
    record("luders_marginals_analytic", d_table < TOL_STRICT, f"maxdiff={d_table:.2e}")
    record("luders_R_analytic", d_R < TOL_STRICT, f"maxdiff={d_R:.2e}")
    record("luders_S_analytic", d_S < TOL_STRICT, f"maxdiff={d_S:.2e}")
    record("luders_bloch_vs_dm", d_dm < TOL_STRICT, f"maxdiff={d_dm:.2e}")

    rng = np.random.default_rng(7)
    big_S = sum(
        abs(luders_qubit_analytics(random_bloch_state(rng), random_bloch_axis(rng), random_bloch_axis(rng))["S"])
        > 0.05
        for _ in range(500)
    )
    record("luders_marginal_reciprocity_nonzero", big_S > 300, f"{big_S}/500 |S|>0.05")


def check_linear_classical(n=1000, seed=202):
    dmax = 0.0
    for i in range(n):
        prng = np.random.default_rng(seed * 997 + i)
        params = dict(
            d=8,
            m_scale=1.0,
            v_scale=float(prng.uniform(0, 1.5)),
            lam=float(prng.uniform(0.3, 0.95)),
            g=1.0,
        )
        t = S.run_2q_classical(params, np.random.default_rng(seed + i), readout="linear")
        res = S.qq_residuals(t)
        cf = S.lin_classical_closed_form(params, np.random.default_rng(seed + i))
        dmax = max(
            dmax,
            abs(res["q_A"] - cf["q_A"]),
            abs(res["q_B"] - cf["q_B"]),
            abs(res["R"] - cf["R"]),
            abs(res["S"] - cf["S"]),
        )
    record("linear_classical_closed_form", dmax < TOL_STRICT, f"maxdiff={dmax:.2e}")


def check_quantum_channels(seed=404):
    rng = np.random.default_rng(seed)
    qmax = 0.0
    for _ in range(50):
        t = S.run_2q_qreservoir(dict(n=3, p=0.0, dynamics_on=False, use_update=True), rng)
        res = S.qq_residuals(t)
        qmax = max(qmax, abs(res["q_A"]), abs(res["q_B"]))
    record("qreservoir_no_dynamics_p0", qmax < TOL_STRICT, f"max|q|={qmax:.2e}")

    dmax = 0.0
    for _ in range(100):
        seed_i = int(rng.integers(0, 2**31))
        params = dict(
            n=int(rng.integers(1, 5)),
            p=float(rng.uniform(0, 1)),
            dynamics_on=bool(rng.integers(0, 2)),
        )
        t1 = S.run_2q_qreservoir(params, np.random.default_rng(seed_i))
        t2 = S.run_2q_qreservoir_selective(params, np.random.default_rng(seed_i))
        dmax = max(
            dmax,
            abs(t1["pA2"] - t2["pA2"]),
            abs(t1["pB2"] - t2["pB2"]),
            abs(t1["pA1"] - t2["pA1"]),
            abs(t1["pB1"] - t2["pB1"]),
        )
    record("selective_vs_nonselective", dmax < TOL_STRICT, f"maxdiff={dmax:.2e}")

    dmax = 0.0
    for _ in range(50):
        n = int(rng.integers(2, 6))
        rho0 = random_pure_rho(n, rng)
        a, b = random_bloch_axis(rng), random_bloch_axis(rng)
        rho1 = dephase_qubit(rho0, n, 0, b, 1.0)
        full = yes_prob(rho1, embed(projector_from_axis(a), 0, n))
        red = yes_prob(dephase_qubit(partial_trace_qubit0(rho0, n), 1, 0, b, 1.0), projector_from_axis(a))
        dmax = max(dmax, abs(full - red))
    record("multiqubit_partial_trace_reduction", dmax < TOL_LOOSE, f"maxdiff={dmax:.2e}")

    dmax = 0.0
    for _ in range(100):
        n = int(rng.integers(1, 5))
        rho0 = random_pure_rho(n, rng) if n > 1 else bloch_to_rho(random_bloch_state(rng))
        a = random_bloch_axis(rng)
        Pa = embed(projector_from_axis(a), 0, n)
        before = yes_prob(rho0, Pa)
        after = 0.0
        for out in (1, 0):
            po = before if out == 1 else 1.0 - before
            after += po * yes_prob(selective_partial_collapse(rho0, n, 0, a, 1.0, out), Pa)
        dmax = max(dmax, abs(before - after))
    record("same_axis_invariance", dmax < TOL_STRICT, f"maxdiff={dmax:.2e}")

    rng2 = np.random.default_rng(seed + 1)
    qmax = 0.0
    for _ in range(200):
        t = S.run_2q_classical(dict(d=10, m_scale=1.0, v_scale=1.0, lam=0.8, g=1e-7), rng2, readout="sigmoid")
        res = S.qq_residuals(t)
        qmax = max(qmax, abs(res["q_A"]), abs(res["q_B"]))
    record("sigmoid_zero_gain_limit", qmax < 1e-5, f"max|q|={qmax:.2e}")

    rng3 = np.random.default_rng(seed + 2)
    ok = True
    for _ in range(30):
        t = S.run_2q_qreservoir(dict(n=4, p=float(rng3.uniform(0, 1)), dynamics_on=True, use_update=True), rng3)
        ok = ok and t["extras"]["rho_valid"]
    record("density_matrix_validity", ok)


def check_unsharp_and_uncertainty(seed=707):
    rng = np.random.default_rng(seed)
    dmax_sym = 0.0
    for _ in range(300):
        n = int(rng.integers(1, 5))
        p = float(rng.uniform(0.0, 1.0))
        t = S.run_2q_qreservoir(dict(n=n, p=p, dynamics_on=False, use_update=True), rng)
        st = S.wb_stats(t)
        dmax_sym = max(dmax_sym, abs(st["q_QQ"]))
    record("symmetric_unsharp_qq_zero", dmax_sym < TOL_STRICT, f"maxdiff={dmax_sym:.2e}")

    dmax_asym = 0.0
    for _ in range(600):
        n = int(rng.integers(1, 5))
        pu, pd = float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0))
        rho0 = random_pure_rho(n, rng)
        a, b = random_bloch_axis(rng), random_bloch_axis(rng)
        c = 0.5 * (1.0 + float(a @ b))
        Pa, Pb = embed(projector_from_axis(a), 0, n), embed(projector_from_axis(b), 0, n)
        pA = yes_prob(rho0, Pa)
        pB = yes_prob(rho0, Pb)
        from quantum_core import asymmetric_partial_collapse

        d_ab = 0.0
        for out, w in ((1, pA), (0, 1.0 - pA)):
            pb = yes_prob(asymmetric_partial_collapse(rho0, n, 0, a, pu, pd, out), Pb)
            d_ab += w * (pb if out == 1 else 1.0 - pb)
        d_ba = 0.0
        for out, w in ((1, pB), (0, 1.0 - pB)):
            pa = yes_prob(asymmetric_partial_collapse(rho0, n, 0, b, pu, pd, out), Pa)
            d_ba += w * (pa if out == 1 else 1.0 - pa)
        pred = (pA - pB) * c * (pu - pd)
        dmax_asym = max(dmax_asym, abs((d_ab - d_ba) - pred))
    record("asymmetric_breaking_law", dmax_asym < TOL_STRICT, f"maxdiff={dmax_asym:.2e}")

    rng2 = np.random.default_rng(seed + 1)
    max_excess = -1.0
    max_oss = 0.0
    ceil = 4.0 / (3.0 * math.sqrt(3.0))
    for _ in range(2000):
        r, a, b = random_bloch_state(rng2), random_bloch_axis(rng2), random_bloch_axis(rng2)
        g = float(a @ b)
        an = luders_qubit_analytics(r, a, b)
        qA, qB = an["q_A"], an["q_B"]
        excess = (qA * qA + qB * qB + 2.0 * g * qA * qB) - ((1.0 - g * g) ** 2) / 4.0
        max_excess = max(max_excess, excess)
        max_oss = max(max_oss, abs(qA) + abs(qB))
    record("uncertainty_ellipse_bound", max_excess <= 1e-14 and max_oss <= ceil + 1e-14, f"max_excess={max_excess:.2e}")


def check_lp_machinery(seed=505):
    rng = np.random.default_rng(seed)
    bat = S.run_3q_classical(dict(d=10, m_scale=1.0, v_scale=0.0, lam=1.0, g=1.5), np.random.default_rng(11), use_sigmoid=True)
    gs = A.global_section_lp(bat["marg"])
    record("lp_nofeedback_feasible", gs["feasible"], f"eps*={gs['eps_star']:.2e}")

    marg_bad = {o + k: (0.9 if o[0] == k else 0.1) for o in ("ABC", "BCA", "CAB", "ACB", "BAC", "CBA") for k in "ABC"}
    gs_bad = A.global_section_lp(marg_bad)
    record("lp_contradictory_infeasible", (not gs_bad["feasible"]) and gs_bad["eps_star"] > 1e-6, f"eps*={gs_bad['eps_star']:.3f}")

    x = rng.dirichlet(np.ones(8))
    marg_real = {}
    atoms = list(itertools.product([1, 0], repeat=3))
    for o in ("ABC", "BCA", "CAB"):
        for qi, k in enumerate("ABC"):
            marg_real[o + k] = sum(x[ai] for ai, atom in enumerate(atoms) if atom[qi] == 1)
    record("lp_real_joint_feasible", A.global_section_lp(marg_real)["feasible"])

    bat2 = S.run_3q_classical(dict(d=10, m_scale=1.0, v_scale=0.0, lam=1.0, g=1.5), np.random.default_rng(13), use_sigmoid=True)
    cbd = A.cbd_lite_lp(bat2["marg"])
    record("lp_cbd_lite_nofeedback", cbd["feasible"], f"eps*={cbd['eps_star']:.2e}")

    bat3 = S.run_3q_qreservoir(dict(n=3, p=0.7, dynamics_on=True, use_update=True), np.random.default_rng(17))
    norm_ok = all(
        abs(sum(v for kk, v in bat3["joint"].items() if kk[:3] == o) - 1.0) < 1e-9
        for o in ("".join(p) for p in itertools.permutations("ABC"))
    )
    record("battery_joints_normalized", norm_ok)

    # Analytic contextual rank-2 counterexample: Gamma = 1 > 0.
    r2_ctx = A.cbd2_lp([0.5, 0.0, 0.0, 0.5], [0.0, 0.5, 0.5, 0.0])
    r2_nonctx = A.cbd2_lp([0.5, 0.0, 0.0, 0.5], [0.5, 0.0, 0.0, 0.5])
    record("rank2_cbd_counterexample", (not r2_ctx["feasible"]) and r2_nonctx["feasible"])


if __name__ == "__main__":
    check_luders()
    check_linear_classical()
    check_quantum_channels()
    check_unsharp_and_uncertainty()
    check_lp_machinery()
    fails = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
    sys.exit(1 if fails else 0)
