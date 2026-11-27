import itertools
import numpy as np

from quantum_core import (
    apply_unitary,
    asymmetric_partial_collapse,
    bloch_to_rho,
    dephase_qubit,
    embed,
    entanglement_entropy_qubit0,
    haar_unitary,
    is_valid_rho,
    luders_qubit_analytics,
    luders_qubit_table,
    projector_from_axis,
    random_bloch_axis,
    random_bloch_state,
    random_pure_rho,
    random_pure_rho_entangled,
    replicability_1q,
    selective_partial_collapse,
    yes_prob,
)


def qq_residuals(t):
    qA = t["pA1"] - t["pA2"]
    qB = t["pB1"] - t["pB2"]
    return dict(q_A=qA, q_B=qB, R=qA - qB, S=qA + qB)


def wb_stats(t, band_q=0.02, band_oss=0.05):
    qA = t["pA1"] - t["pA2"]
    qB = t["pB1"] - t["pB2"]
    dAB = t["joint_AB"][0] + t["joint_AB"][3]
    dBA = t["joint_BA"][0] + t["joint_BA"][3]
    qQQ = dAB - dBA
    oss = abs(qA) + abs(qB)
    a_rate = t["joint_AB"][1] / t["pA1"] if t["pA1"] > 1e-12 else float("nan")
    b_rate = t["joint_AB"][2] / (1.0 - t["pA1"]) if t["pA1"] < 1.0 - 1e-12 else float("nan")
    return dict(
        q_A=qA,
        q_B=qB,
        R=qA - qB,
        S=qA + qB,
        q_QQ=qQQ,
        OSS=oss,
        Gamma=abs(qQQ) - oss,
        D_AB=dAB,
        D_BA=dBA,
        disagree_AB=t["joint_AB"][1] + t["joint_AB"][2],
        disagree_BA=t["joint_BA"][1] + t["joint_BA"][2],
        a_rate=a_rate,
        b_rate=b_rate,
        grain=a_rate - b_rate,
        in_human_band=bool(abs(qQQ) <= band_q and oss >= band_oss),
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def random_unit_vector(rng, d):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def _classical_2q_eval(s0, mA, mB, cA, cB, v, lam, read_fn):
    pA1 = read_fn(s0, mA, cA)
    pB1 = read_fn(s0, mB, cB)
    s_p, s_m = lam * s0 + v, lam * s0 - v
    pyB_yes, pyB_no = read_fn(s_p, mB, cB), read_fn(s_m, mB, cB)
    pyA_yes, pyA_no = read_fn(s_p, mA, cA), read_fn(s_m, mA, cA)
    pA2 = pB1 * pyA_yes + (1.0 - pB1) * pyA_no
    pB2 = pA1 * pyB_yes + (1.0 - pA1) * pyB_no
    # Joint cell ordering: [yy, yn, ny, nn].
    jAB = [pA1 * pyB_yes, pA1 * (1.0 - pyB_yes), (1.0 - pA1) * pyB_no, (1.0 - pA1) * (1.0 - pyB_no)]
    jBA = [pB1 * pyA_yes, pB1 * (1.0 - pyA_yes), (1.0 - pB1) * pyA_no, (1.0 - pB1) * (1.0 - pyA_no)]
    rA = pA1 * pyA_yes + (1.0 - pA1) * (1.0 - pyA_no)
    rB = pB1 * pyB_yes + (1.0 - pB1) * (1.0 - pyB_no)
    return dict(
        pA1=pA1,
        pB1=pB1,
        pA2=pA2,
        pB2=pB2,
        joint_AB=jAB,
        joint_BA=jBA,
        R_rep_A=rA,
        R_rep_B=rB,
        extras=dict(s0_norm=float(np.linalg.norm(s0))),
    )


def run_2q_classical(params, rng, readout="sigmoid"):
    d = params["d"]
    s0 = rng.normal(size=d) / np.sqrt(d)
    mA = random_unit_vector(rng, d) * params["m_scale"]
    mB = random_unit_vector(rng, d) * params["m_scale"]
    cA, cB = rng.uniform(-0.5, 0.5, size=2)
    v = random_unit_vector(rng, d) * params["v_scale"]
    lam = params["lam"]
    g = params.get("g", 1.0)

    def read_fn(s, m, c):
        x = m @ s + c
        return float(sigmoid(g * x)) if readout == "sigmoid" else float(x)

    return _classical_2q_eval(s0, mA, mB, cA, cB, v, lam, read_fn)


def run_2q_lin2(params, rng):
    d = params["d"]
    s0 = rng.normal(size=d) / np.sqrt(d)
    mA = random_unit_vector(rng, d) * params["m_scale"]
    mB = random_unit_vector(rng, d) * params["m_scale"]
    pA_t = float(rng.uniform(0.15, 0.85))
    pB_t = float(rng.uniform(0.15, 0.85))
    cA = pA_t - mA @ s0
    cB = pB_t - mB @ s0
    v = random_unit_vector(rng, d) * params["v_scale"]
    lam = params["lam"]
    return _classical_2q_eval(s0, mA, mB, cA, cB, v, lam, lambda s, m, c: float(m @ s + c))


def lin_classical_closed_form(params, rng):
    d = params["d"]
    s0 = rng.normal(size=d) / np.sqrt(d)
    mA = random_unit_vector(rng, d) * params["m_scale"]
    mB = random_unit_vector(rng, d) * params["m_scale"]
    cA, cB = rng.uniform(-0.5, 0.5, size=2)
    v = random_unit_vector(rng, d) * params["v_scale"]
    lam = params["lam"]
    pA1 = mA @ s0 + cA
    pB1 = mB @ s0 + cB
    qA = (1.0 - lam) * (mA @ s0) - (mA @ v) * (2.0 * pB1 - 1.0)
    qB = (1.0 - lam) * (mB @ s0) - (mB @ v) * (2.0 * pA1 - 1.0)
    return dict(q_A=float(qA), q_B=float(qB), R=float(qA - qB), S=float(qA + qB))


def run_3q_classical(params, rng, use_sigmoid=True):
    d = params["d"]
    s0 = rng.normal(size=d) / np.sqrt(d)
    questions = {
        k: (random_unit_vector(rng, d) * params["m_scale"], c)
        for k, c in zip("ABC", rng.uniform(-0.5, 0.5, size=3))
    }
    v = random_unit_vector(rng, d) * params["v_scale"]
    lam = params["lam"]
    g = params.get("g", 1.0)

    def read_fn(s, k):
        m, c = questions[k]
        x = m @ s + c
        return float(sigmoid(g * x)) if use_sigmoid else float(x)

    marg, joint = {}, {}
    for order in itertools.permutations("ABC"):
        order_s = "".join(order)
        for bits in itertools.product([1.0, -1.0], repeat=3):
            s = s0.copy()
            w = 1.0
            for i, k in enumerate(order):
                p = read_fn(s, k)
                w *= p if bits[i] > 0 else (1.0 - p)
                s = lam * s + v * bits[i]
            joint[order_s + "".join("Y" if b > 0 else "N" for b in bits)] = w
        for i, k in enumerate(order):
            marg[order_s + k] = sum(
                w for outcome_s, w in joint.items() if outcome_s[:3] == order_s and outcome_s[3 + i] == "Y"
            )
    return dict(marg=marg, joint=joint)


def run_3q_qreservoir(params, rng):
    n = params["n"]
    p = params["p"]
    dyn = params.get("dynamics_on", True)
    upd = params.get("use_update", True)
    dim = 2 ** n
    rho0 = random_pure_rho(n, rng)
    ax = {k: random_bloch_axis(rng) for k in "ABC"}
    U = {k: (haar_unitary(dim, rng) if dyn else None) for k in "ABC"}
    proj = {k: embed(projector_from_axis(ax[k]), 0, n) for k in "ABC"}
    marg, joint = {}, {}
    for order in itertools.permutations("ABC"):
        order_s = "".join(order)
        for bits in itertools.product([1, 0], repeat=3):
            rho = rho0
            w = 1.0
            for i, k in enumerate(order):
                if dyn:
                    rho = apply_unitary(rho, U[k])
                pk = yes_prob(rho, proj[k])
                w *= pk if bits[i] == 1 else (1.0 - pk)
                if upd:
                    rho = selective_partial_collapse(rho, n, 0, ax[k], p, bits[i])
            joint[order_s + "".join("Y" if b else "N" for b in bits)] = w
        for i, k in enumerate(order):
            marg[order_s + k] = sum(
                w for outcome_s, w in joint.items() if outcome_s[:3] == order_s and outcome_s[3 + i] == "Y"
            )
    return dict(marg=marg, joint=joint)


def luders_qubit_joints(r, a, b):
    r, a, b = map(np.asarray, (r, a, b))
    c = 0.5 * (1.0 + a @ b)
    pA1 = 0.5 * (1.0 + a @ r)
    pB1 = 0.5 * (1.0 + b @ r)
    jAB = [pA1 * c, pA1 * (1.0 - c), (1.0 - pA1) * (1.0 - c), (1.0 - pA1) * c]
    jBA = [pB1 * c, pB1 * (1.0 - c), (1.0 - pB1) * (1.0 - c), (1.0 - pB1) * c]
    return jAB, jBA


def run_2q_luders(params, rng):
    r = random_bloch_state(rng)
    a = random_bloch_axis(rng)
    b = random_bloch_axis(rng)
    t = luders_qubit_table(r, a, b)
    jAB, jBA = luders_qubit_joints(r, a, b)
    an = luders_qubit_analytics(r, a, b)
    t["joint_AB"] = jAB
    t["joint_BA"] = jBA
    t["R_rep_A"] = 1.0
    t["R_rep_B"] = 1.0
    t["extras"] = dict(
        analytic_qA=an["q_A"],
        analytic_qB=an["q_B"],
        analytic_R=an["R"],
        analytic_S=an["S"],
        S0=0.0,
    )
    return t


def run_2q_luders_dm(params, rng):
    r = random_bloch_state(rng)
    a = random_bloch_axis(rng)
    b = random_bloch_axis(rng)
    rho = bloch_to_rho(r)
    Pa, Pb = projector_from_axis(a), projector_from_axis(b)
    pA1 = yes_prob(rho, Pa)
    pB1 = yes_prob(rho, Pb)
    pA2 = yes_prob(dephase_qubit(rho, 1, 0, b, 1.0), Pa)
    pB2 = yes_prob(dephase_qubit(rho, 1, 0, a, 1.0), Pb)
    return dict(pA1=pA1, pB1=pB1, pA2=pA2, pB2=pB2, extras={})


def run_2q_qreservoir(params, rng):
    n = params["n"]
    p_up = params.get("p_up", params.get("p", 1.0))
    p_dn = params.get("p_dn", params.get("p", 1.0))
    dyn = params.get("dynamics_on", True)
    upd = params.get("use_update", True)
    mode = params.get("init_mode", "haar")
    dim = 2 ** n

    rho0 = random_pure_rho_entangled(n, rng, mode)
    ax = {k: random_bloch_axis(rng) for k in "AB"}
    U = {k: (haar_unitary(dim, rng) if dyn else None) for k in "AB"}
    Pa = embed(projector_from_axis(ax["A"]), 0, n)
    Pb = embed(projector_from_axis(ax["B"]), 0, n)

    rhoA_pre = apply_unitary(rho0, U["A"]) if dyn else rho0
    rhoB_pre = apply_unitary(rho0, U["B"]) if dyn else rho0
    pA1 = yes_prob(rhoA_pre, Pa)
    pB1 = yes_prob(rhoB_pre, Pb)

    def backact(rho, k, out):
        if not upd:
            return rho
        return asymmetric_partial_collapse(rho, n, 0, ax[k], p_up, p_dn, out)

    jAB, jBA = [0.0] * 4, [0.0] * 4
    rho_ok = True
    for out, w in ((1, pA1), (0, 1.0 - pA1)):
        rp = backact(rhoA_pre, "A", out)
        r1 = apply_unitary(rp, U["B"]) if dyn else rp
        rho_ok = rho_ok and is_valid_rho(r1)
        pb = yes_prob(r1, Pb)
        jAB[0 if out else 2] += w * pb
        jAB[1 if out else 3] += w * (1.0 - pb)
    for out, w in ((1, pB1), (0, 1.0 - pB1)):
        rp = backact(rhoB_pre, "B", out)
        r1 = apply_unitary(rp, U["A"]) if dyn else rp
        rho_ok = rho_ok and is_valid_rho(r1)
        pa = yes_prob(r1, Pa)
        jBA[0 if out else 2] += w * pa
        jBA[1 if out else 3] += w * (1.0 - pa)

    pA2 = jBA[0] + jBA[2]
    pB2 = jAB[0] + jAB[2]
    rA = replicability_1q(rhoA_pre, n, 0, ax["A"], p_up if upd else 0.0)
    rB = replicability_1q(rhoB_pre, n, 0, ax["B"], p_up if upd else 0.0)
    S0 = entanglement_entropy_qubit0(rho0, n)
    return dict(
        pA1=pA1,
        pB1=pB1,
        pA2=pA2,
        pB2=pB2,
        joint_AB=jAB,
        joint_BA=jBA,
        R_rep_A=rA,
        R_rep_B=rB,
        extras=dict(rho_valid=bool(rho_ok), S0=S0),
    )


def run_2q_qreservoir_selective(params, rng):
    n = params["n"]
    p = params["p"]
    dyn = params.get("dynamics_on", True)
    dim = 2 ** n
    rho0 = random_pure_rho(n, rng)
    ax = {k: random_bloch_axis(rng) for k in "AB"}
    U = {k: (haar_unitary(dim, rng) if dyn else None) for k in "AB"}
    Pa = embed(projector_from_axis(ax["A"]), 0, n)
    Pb = embed(projector_from_axis(ax["B"]), 0, n)

    rhoA = apply_unitary(rho0, U["A"]) if dyn else rho0
    rhoB = apply_unitary(rho0, U["B"]) if dyn else rho0
    pA1 = yes_prob(rhoA, Pa)
    pB1 = yes_prob(rhoB, Pb)

    pA2 = 0.0
    for out in (1, 0):
        rp = selective_partial_collapse(rhoB, n, 0, ax["B"], p, out)
        r1 = apply_unitary(rp, U["A"]) if dyn else rp
        pA2 += (pB1 if out == 1 else 1.0 - pB1) * yes_prob(r1, Pa)

    pB2 = 0.0
    for out in (1, 0):
        rp = selective_partial_collapse(rhoA, n, 0, ax["A"], p, out)
        r1 = apply_unitary(rp, U["B"]) if dyn else rp
        pB2 += (pA1 if out == 1 else 1.0 - pA1) * yes_prob(r1, Pb)

    return dict(pA1=pA1, pB1=pB1, pA2=pA2, pB2=pB2, extras={})


def retune_quantum(params, rng):
    n = params["n"]
    p = params["p"]
    dyn = params.get("dynamics_on", True)
    upd = params.get("use_update", True)
    K = int(params.get("K", 20))
    dim = 2 ** n
    rho0 = random_pure_rho(n, rng)
    U = {k: (haar_unitary(dim, rng) if dyn else None) for k in "AB"}
    hits = 0
    for _ in range(K):
        ax = {k: random_bloch_axis(rng) for k in "AB"}
        Pa = embed(projector_from_axis(ax["A"]), 0, n)
        Pb = embed(projector_from_axis(ax["B"]), 0, n)
        rhoA_pre = apply_unitary(rho0, U["A"]) if dyn else rho0
        rhoB_pre = apply_unitary(rho0, U["B"]) if dyn else rho0
        pA1 = yes_prob(rhoA_pre, Pa)
        pB1 = yes_prob(rhoB_pre, Pb)
        jAB, jBA = [0.0] * 4, [0.0] * 4
        for out, w in ((1, pA1), (0, 1.0 - pA1)):
            rp = selective_partial_collapse(rhoA_pre, n, 0, ax["A"], p, out) if upd else rhoA_pre
            r1 = apply_unitary(rp, U["B"]) if dyn else rp
            pb = yes_prob(r1, Pb)
            jAB[0 if out else 2] += w * pb
            jAB[1 if out else 3] += w * (1.0 - pb)
        for out, w in ((1, pB1), (0, 1.0 - pB1)):
            rp = selective_partial_collapse(rhoB_pre, n, 0, ax["B"], p, out) if upd else rhoB_pre
            r1 = apply_unitary(rp, U["A"]) if dyn else rp
            pa = yes_prob(r1, Pa)
            jBA[0 if out else 2] += w * pa
            jBA[1 if out else 3] += w * (1.0 - pa)
        st = wb_stats(
            dict(
                pA1=pA1,
                pB1=pB1,
                pA2=jBA[0] + jBA[2],
                pB2=jAB[0] + jAB[2],
                joint_AB=jAB,
                joint_BA=jBA,
            )
        )
        if st["in_human_band"]:
            hits += 1
    return hits / K


def retune_classical(params, rng, readout="sigmoid"):
    d = params["d"]
    s0 = rng.normal(size=d) / np.sqrt(d)
    v = random_unit_vector(rng, d) * params["v_scale"]
    lam = params["lam"]
    g = params.get("g", 1.0)
    K = int(params.get("K", 20))
    hits = 0

    def read_fn(s, m, c):
        x = m @ s + c
        return float(sigmoid(g * x)) if readout == "sigmoid" else float(x)

    for _ in range(K):
        mA = random_unit_vector(rng, d) * params["m_scale"]
        mB = random_unit_vector(rng, d) * params["m_scale"]
        cA, cB = rng.uniform(-0.5, 0.5, size=2)
        t = _classical_2q_eval(s0, mA, mB, cA, cB, v, lam, read_fn)
        if wb_stats(t)["in_human_band"]:
            hits += 1
    return hits / K


SUBSTRATES_2Q = {
    "luders_qubit": run_2q_luders,
    "luders_qubit_dm": run_2q_luders_dm,
    "lin_classical": lambda params, rng: run_2q_classical(params, rng, readout="linear"),
    "nl_classical": lambda params, rng: run_2q_classical(params, rng, readout="sigmoid"),
    "qreservoir": run_2q_qreservoir,
    "qreservoir_selective": run_2q_qreservoir_selective,
}


def run_3q(substrate, params, rng):
    if substrate == "qreservoir":
        return run_3q_qreservoir(params, rng)
    if substrate in ("lin_classical", "nl_classical"):
        return run_3q_classical(params, rng, use_sigmoid=(substrate == "nl_classical"))
    raise ValueError(f"Unsupported 3q substrate: {substrate}")


def instance_params(substrate, rng, overrides=None):
    if substrate in ("lin_classical", "nl_classical"):
        p = dict(
            d=50,
            m_scale=float(rng.uniform(0.5, 1.5)),
            v_scale=float(rng.uniform(0.0, 1.5)),
            lam=float(rng.uniform(0.3, 0.95)),
            g=float(rng.choice([0.5, 1.0, 2.0, 4.0])),
            use_sigmoid=(substrate == "nl_classical"),
        )
    elif substrate == "qreservoir":
        p = dict(n=int(rng.integers(3, 7)), p=float(rng.uniform(0.0, 1.0)), dynamics_on=True, use_update=True)
    else:
        p = dict()
    if overrides:
        p.update(overrides)
    return p
