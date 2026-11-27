import itertools
import sys
import time
from pathlib import Path

import numpy as np

import analysis as A
import substrates as S
from quantum_core import random_bloch_axis, random_bloch_state

BASE_DIR = Path(__file__).resolve().parent
LEDGER_DIR = BASE_DIR / "results" / "ledger"
BAND_Q = 0.02
BAND_OSS = 0.05


def ledger_path(exp):
    return LEDGER_DIR / f"{exp}.jsonl"


def done_ids(path):
    done = set()
    if path.exists():
        for r in A.read_ledger(path):
            done.add(r["instance_id"])
    return done


def run_block(exp, substrate, n_instances, base_seed, cell="main", param_overrides=None):
    path = ledger_path(exp)
    done = done_ids(path)
    led = A.Ledger(path)
    n_new = 0
    t0 = time.time()
    for i in range(n_instances):
        iid = f"{exp}_{substrate}_{cell}_{i:06d}"
        if iid in done:
            continue
        rng = np.random.default_rng(base_seed + i)
        params = S.instance_params(substrate, rng, param_overrides)
        t = S.SUBSTRATES_2Q[substrate](params, rng)
        row = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
        row["extras"] = {k: (v if isinstance(v, (int, float, bool)) else str(v)) for k, v in t.get("extras", {}).items()}
        row["joint_AB"] = [float(x) for x in t["joint_AB"]]
        row["joint_BA"] = [float(x) for x in t["joint_BA"]]
        row["instance_id"] = iid
        row["substrate"] = substrate
        row["cell"] = cell
        row["params"] = {k: v for k, v in params.items() if isinstance(v, (int, float, bool))}
        row["seed"] = base_seed + i
        led.write(row)
        n_new += 1
    print(f"[{exp}/{substrate}/{cell}] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


def run_e2(exp="E2"):
    configs = [
        ("qreservoir", 300, 211000, "q_p1_full", dict(n=4, p=1.0, dynamics_on=True, use_update=True)),
        ("qreservoir", 300, 212000, "q_p0_noupd", dict(n=4, p=0.0, dynamics_on=True, use_update=False)),
        ("qreservoir", 300, 213000, "q_p1_nodyn", dict(n=4, p=1.0, dynamics_on=False, use_update=True)),
        ("nl_classical", 300, 214000, "c_fb", dict(g=1.0)),
        ("lin_classical", 300, 225000, "c_lin_generic", None),
        ("lin_classical", 300, 226000, "c_lin_nodisturb", dict(lam=1.0, v_scale=0.0)),
    ]
    path = ledger_path(exp)
    done = done_ids(path)
    led = A.Ledger(path)
    for substrate, n_instances, base_seed, cell, param_overrides in configs:
        t0 = time.time()
        n_new = 0
        for i in range(n_instances):
            iid = f"{exp}_{substrate}_{cell}_{i:06d}"
            if iid in done:
                continue
            rng = np.random.default_rng(base_seed + i)
            params = S.instance_params(substrate, rng, param_overrides)
            for _ in range(50):
                bat = S.run_3q(substrate, params, rng)
                if all(-1e-9 <= v <= 1.0 + 1e-9 for v in bat["marg"].values()):
                    break
            else:
                continue
            gs = A.global_section_lp(bat["marg"])
            cbd = A.cbd_lite_lp(bat["marg"])
            firstpos = {q: bat["marg"][q + "".join(x for x in "ABC" if x != q) + q] for q in "ABC"}
            dist = [
                abs(bat["marg"]["".join(o) + q] - firstpos[q])
                for o in itertools.permutations("ABC")
                for q in o
            ]
            led.write(
                dict(
                    instance_id=iid,
                    substrate=substrate,
                    cell=cell,
                    params={k: v for k, v in params.items() if isinstance(v, (int, float, bool))},
                    seed=base_seed + i,
                    lp_feasible=bool(gs["feasible"]),
                    eps_star=float(gs["eps_star"]),
                    cert_hash=A.hash_certificate(gs["certificate"]),
                    cbd_feasible=bool(cbd["feasible"]),
                    cbd_eps=float(cbd["eps_star"]),
                    n_atoms=cbd["n_atoms"],
                    marg={k: round(v, 10) for k, v in bat["marg"].items()},
                    firstpos={k: round(v, 10) for k, v in firstpos.items()},
                    disturbance_mag=float(np.mean(dist)),
                )
            )
            n_new += 1
        print(f"[{exp}/{substrate}/{cell}] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


def run_e4(exp="E4", base_seed=90400):
    p_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
    path = ledger_path(exp)
    done = done_ids(path)
    led = A.Ledger(path)
    t0 = time.time()
    n_new = 0
    for p in p_grid:
        for i in range(500):
            iid = f"{exp}_qres_p{p:.2f}_{i:06d}"
            if iid in done:
                continue
            seed = base_seed + int(p * 1000) + i
            params = dict(n=4, p=p, dynamics_on=True, use_update=True)
            t = S.run_2q_qreservoir(params, np.random.default_rng(seed))
            row = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
            row["joint_AB"] = [float(x) for x in t["joint_AB"]]
            row["joint_BA"] = [float(x) for x in t["joint_BA"]]
            row.update(instance_id=iid, substrate="qreservoir", cell=f"p{p:.2f}", params=params, seed=seed)
            led.write(row)
            n_new += 1
        for i in range(300):
            iid = f"{exp}_qresoff_p{p:.2f}_{i:06d}"
            if iid in done:
                continue
            seed = base_seed + int(p * 1000) + 5555 + i
            params = dict(n=4, p=p, dynamics_on=False, use_update=True)
            t = S.run_2q_qreservoir(params, np.random.default_rng(seed))
            row = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
            row["joint_AB"] = [float(x) for x in t["joint_AB"]]
            row["joint_BA"] = [float(x) for x in t["joint_BA"]]
            row.update(instance_id=iid, substrate="qreservoir", cell=f"nodyn_p{p:.2f}", params=params, seed=seed)
            led.write(row)
            n_new += 1
    for vs in [0.0, 0.25, 0.5, 1.0, 2.0]:
        for i in range(500):
            iid = f"{exp}_nlc_v{vs:.2f}_{i:06d}"
            if iid in done:
                continue
            seed = base_seed + int(vs * 1000) + 7777 + i
            params = dict(d=50, m_scale=1.0, v_scale=vs, lam=0.8, g=1.0)
            t = S.run_2q_classical(params, np.random.default_rng(seed), readout="sigmoid")
            row = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
            row["joint_AB"] = [float(x) for x in t["joint_AB"]]
            row["joint_BA"] = [float(x) for x in t["joint_BA"]]
            row.update(instance_id=iid, substrate="nl_classical", cell=f"v{vs:.2f}", params=params, seed=seed)
            led.write(row)
            n_new += 1
    print(f"[{exp}] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


def run_e6(n_instances=300, base_seed=310000):
    path = ledger_path("E6")
    done = done_ids(path)
    led = A.Ledger(path)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_new = 0
    t0 = time.time()
    for dyn in (False, True):
        cells = [(pu, pd) for pu in grid for pd in grid] if not dyn else [(1.0, 0.0), (1.0, 0.5), (0.5, 0.0)]
        for pu, pd in cells:
            for n in ([1, 4] if not dyn else [4]):
                for i in range(n_instances):
                    iid = f"E6_qres_dyn{int(dyn)}_n{n}_pu{pu:.2f}_pd{pd:.2f}_{i:06d}"
                    if iid in done:
                        continue
                    seed = base_seed + int(pu * 100) * 131 + int(pd * 100) * 17 + (0 if n == 1 else 7) * 1000003 + i
                    params = dict(n=n, p_up=pu, p_dn=pd, dynamics_on=dyn, use_update=True, init_mode="haar")
                    t = S.run_2q_qreservoir(params, np.random.default_rng(seed))
                    st = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
                    st.update(
                        instance_id=iid,
                        substrate="qreservoir",
                        cell=f"dyn{int(dyn)}_n{n}_pu{pu:.2f}_pd{pd:.2f}",
                        params=params,
                        seed=base_seed + i,
                        joint_AB=[float(x) for x in t["joint_AB"]],
                        joint_BA=[float(x) for x in t["joint_BA"]],
                        R_rep_A=float(t["R_rep_A"]),
                    )
                    led.write(st)
                    n_new += 1
    print(f"[E6] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


def run_e7(n_instances=400, base_seed=320000):
    path = ledger_path("E7")
    done = done_ids(path)
    led = A.Ledger(path)
    p_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_new = 0
    t0 = time.time()
    specs = (
        [("qres_dynoff", dict(n=4, dynamics_on=False, use_update=True), p) for p in p_grid]
        + [("qres_dynon", dict(n=4, dynamics_on=True, use_update=True), p) for p in p_grid]
        + [("nl_classical", None, p) for p in p_grid]
        + [("lin2_classical", None, p) for p in p_grid]
        + [("luders_qubit", None, p) for p in p_grid]
    )
    for name, base_params, p in specs:
        for i in range(n_instances):
            iid = f"E7_{name}_p{p:.2f}_{i:06d}"
            if iid in done:
                continue
            rng = np.random.default_rng(base_seed + int(p * 100) * 211 + i)
            if name.startswith("qres"):
                params = dict(base_params)
                params.update(dict(p_up=p, p_dn=p, n=base_params["n"], init_mode="haar", use_update=base_params["use_update"] and p > 0))
                t = S.run_2q_qreservoir(params, rng)
            elif name == "nl_classical":
                params = dict(
                    d=50,
                    m_scale=float(rng.uniform(0.5, 1.5)),
                    v_scale=float(rng.uniform(0.0, 1.5)),
                    lam=float(rng.uniform(0.3, 0.95)),
                    g=float(rng.choice([0.5, 1.0, 2.0, 4.0])),
                )
                t = S.run_2q_classical(params, rng, readout="sigmoid")
            elif name == "lin2_classical":
                params = dict(
                    d=50,
                    m_scale=float(rng.uniform(0.4, 0.9)),
                    v_scale=float(rng.uniform(0.0, 0.8)),
                    lam=float(rng.uniform(0.3, 0.95)),
                )
                t = S.run_2q_lin2(params, rng)
                if not A.joints_proper(t["joint_AB"], t["joint_BA"]):
                    continue
            else:
                params = dict(p=p)
                t = S.run_2q_luders(params, rng)
            st = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
            st.update(
                instance_id=iid,
                substrate=name,
                cell=f"{name}_p{p:.2f}",
                params=params,
                seed=base_seed + i,
                joint_AB=[float(x) for x in t["joint_AB"]],
                joint_BA=[float(x) for x in t["joint_BA"]],
                R_rep_A=float(t["R_rep_A"]),
                R_rep_B=float(t["R_rep_B"]),
                S0=float(t.get("extras", {}).get("S0", float("nan"))),
            )
            led.write(st)
            n_new += 1
    print(f"[E7] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


def run_e8(n_instances=500, base_seed=330000):
    path = ledger_path("E8")
    done = done_ids(path)
    led = A.Ledger(path)
    n_new = 0
    t0 = time.time()
    mode_offsets = {"product": 0, "maxpair": 1, "haar": 2}
    for mode in ("product", "maxpair", "haar"):
        for i in range(n_instances):
            iid = f"E8_qres_{mode}_{i:06d}"
            if iid in done:
                continue
            seed = base_seed + mode_offsets[mode] * 1000003 + i
            params = dict(n=4, p_up=1.0, p_dn=1.0, dynamics_on=True, use_update=True, init_mode=mode)
            t = S.run_2q_qreservoir(params, np.random.default_rng(seed))
            st = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
            st.update(
                instance_id=iid,
                substrate="qreservoir",
                cell=mode,
                params=params,
                seed=base_seed + i,
                S0=float(t["extras"]["S0"]),
            )
            led.write(st)
            n_new += 1
    print(f"[E8] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


def run_e1b(n_nl=1000, n_qr=500, n_qr_dyn=300, base_seed=340000):
    path = ledger_path("E1b")
    done = done_ids(path)
    led = A.Ledger(path)
    n_new = 0
    t0 = time.time()
    for i in range(n_nl):
        iid = f"E1b_nl_classical_{i:06d}"
        if iid in done:
            continue
        rng = np.random.default_rng(base_seed + i)
        params = dict(
            d=50,
            m_scale=float(rng.uniform(0.5, 1.5)),
            v_scale=float(rng.uniform(0.0, 1.5)),
            lam=float(rng.uniform(0.3, 0.95)),
            g=float(rng.choice([0.5, 1.0, 2.0, 4.0])),
            K=20,
        )
        frac = S.retune_classical(params, rng, readout="sigmoid")
        led.write(dict(instance_id=iid, substrate="nl_classical", cell="retune_classical", params=params, seed=base_seed + i, tunable_fraction=frac))
        n_new += 1
    for i in range(n_qr):
        iid = f"E1b_qres_dynoff_{i:06d}"
        if iid in done:
            continue
        rng = np.random.default_rng(base_seed + 1000000 + i)
        params = dict(n=int(rng.integers(3, 7)), p=float(rng.uniform(0.2, 1.0)), dynamics_on=False, use_update=True, K=20)
        frac = S.retune_quantum(params, rng)
        led.write(dict(instance_id=iid, substrate="qreservoir", cell="retune_qres_dynoff", params=params, seed=base_seed + 1000000 + i, tunable_fraction=frac))
        n_new += 1
    for i in range(n_qr_dyn):
        iid = f"E1b_qres_dynon_{i:06d}"
        if iid in done:
            continue
        rng = np.random.default_rng(base_seed + 2000000 + i)
        params = dict(n=4, p=1.0, dynamics_on=True, use_update=True, K=20)
        frac = S.retune_quantum(params, rng)
        led.write(dict(instance_id=iid, substrate="qreservoir", cell="retune_qres_dynon", params=params, seed=base_seed + 2000000 + i, tunable_fraction=frac))
        n_new += 1
    print(f"[E1b] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


def run_e1c(n_instances=2000, base_seed=350000):
    path = ledger_path("E1c")
    done = done_ids(path)
    led = A.Ledger(path)
    n_new = n_rej = 0
    t0 = time.time()
    for i in range(n_instances):
        iid = f"E1c_lin2_classical_{i:06d}"
        if iid in done:
            continue
        rng = np.random.default_rng(base_seed + i)
        params = dict(
            d=50,
            m_scale=float(rng.uniform(0.4, 0.9)),
            v_scale=float(rng.uniform(0.0, 0.8)),
            lam=float(rng.uniform(0.3, 0.95)),
        )
        t = S.run_2q_lin2(params, rng)
        if not A.joints_proper(t["joint_AB"], t["joint_BA"]):
            n_rej += 1
            continue
        st = S.wb_stats(t, band_q=BAND_Q, band_oss=BAND_OSS)
        st.update(
            instance_id=iid,
            substrate="lin2_classical",
            cell="main",
            params=params,
            seed=base_seed + i,
            joint_AB=[float(x) for x in t["joint_AB"]],
            joint_BA=[float(x) for x in t["joint_BA"]],
            R_rep_A=float(t["R_rep_A"]),
        )
        led.write(st)
        n_new += 1
    print(f"[E1c] +{n_new} rows, {n_rej} rejected ({time.time() - t0:.1f}s)", flush=True)


def run_e12():
    path = ledger_path("E12")
    done = done_ids(path)
    led = A.Ledger(path)
    t0 = time.time()
    n_new = 0
    for vs, g in [(0.5, 1.0), (1.0, 1.0), (1.5, 2.0), (2.0, 4.0)]:
        iid = f"E12_classical_v{vs}_g{g}"
        if iid in done:
            continue
        seed = 0
        K = 8000
        d = 30
        lam = 0.8
        rng = np.random.default_rng(seed)
        s0 = rng.normal(size=d) / np.sqrt(d)
        v = S.random_unit_vector(rng, d) * vs
        best, best_cfg = 0.0, None
        for k in range(K):
            rng_k = np.random.default_rng(hash((seed, k)) % 2**31)
            mA = S.random_unit_vector(rng_k, d)
            mB = S.random_unit_vector(rng_k, d)
            cA, cB = rng_k.uniform(-2, 2, size=2)
            read = lambda s, m, c: float(S.sigmoid(g * (m @ s + c)))
            pA1, pB1 = read(s0, mA, cA), read(s0, mB, cB)
            pA2 = pB1 * read(lam * s0 + v, mA, cA) + (1.0 - pB1) * read(lam * s0 - v, mA, cA)
            pB2 = pA1 * read(lam * s0 + v, mB, cB) + (1.0 - pA1) * read(lam * s0 - v, mB, cB)
            oss = abs(pA1 - pA2) + abs(pB1 - pB2)
            if oss > best:
                best = oss
                best_cfg = [pA1, pA2, pB1, pB2]
        led.write(
            dict(
                instance_id=iid,
                substrate="nl_classical",
                cell="adversarial_readout",
                params=dict(v_scale=vs, g=g, K=K, lam=lam, d=d, seed=seed),
                max_OSS=best,
                best_table=best_cfg,
            )
        )
        n_new += 1

    iid_q = "E12_luders_fully_adversarial"
    if iid_q not in done:
        seed = 1
        K = 50000
        best = 0.0
        for k in range(K):
            rng_k = np.random.default_rng(hash((seed, k)) % 2**31)
            t = S.luders_qubit_table(random_bloch_state(rng_k), random_bloch_axis(rng_k), random_bloch_axis(rng_k))
            best = max(best, abs(t["pA1"] - t["pA2"]) + abs(t["pB1"] - t["pB2"]))
        led.write(
            dict(
                instance_id=iid_q,
                substrate="luders_qubit",
                cell="adversarial_full",
                params=dict(K=K, seed=seed),
                max_OSS=best,
            )
        )
        n_new += 1
    print(f"[E12] +{n_new} rows ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "E1"):
        run_block("E1", "luders_qubit", 2000, 111000, cell="main")
        run_block("E1", "lin_classical", 2000, 112000, cell="main")
        run_block("E1", "nl_classical", 2000, 113000, cell="main")
        run_block("E1", "qreservoir", 2000, 114000, cell="main")
    if which in ("all", "E5"):
        run_block("E5", "qreservoir", 1000, 521000, cell="q_dyn_upd", param_overrides=dict(n=4, p=1.0, dynamics_on=True, use_update=True))
        run_block("E5", "qreservoir", 1000, 522000, cell="q_dyn_noupd", param_overrides=dict(n=4, p=1.0, dynamics_on=True, use_update=False))
        run_block("E5", "qreservoir", 1000, 523000, cell="q_nodyn_upd", param_overrides=dict(n=4, p=1.0, dynamics_on=False, use_update=True))
        run_block("E5", "qreservoir", 1000, 524000, cell="q_nodyn_noupd", param_overrides=dict(n=4, p=1.0, dynamics_on=False, use_update=False))
        run_block("E5", "nl_classical", 1000, 531000, cell="c_fade_fb", param_overrides=dict(lam=0.8, g=1.0))
        run_block("E5", "nl_classical", 1000, 532000, cell="c_fade_only", param_overrides=dict(lam=0.8, g=1.0, v_scale=0.0))
        run_block("E5", "nl_classical", 1000, 533000, cell="c_fb_only", param_overrides=dict(lam=1.0, g=1.0))
        run_block("E5", "nl_classical", 1000, 534000, cell="c_neither", param_overrides=dict(lam=1.0, g=1.0, v_scale=0.0))
    if which in ("all", "E4"):
        run_e4()
    if which in ("all", "E2"):
        run_e2()
    if which in ("all", "E6"):
        run_e6()
    if which in ("all", "E7"):
        run_e7()
    if which in ("all", "E8"):
        run_e8()
    if which in ("all", "E1b"):
        run_e1b()
    if which in ("all", "E1c"):
        run_e1c()
    if which in ("all", "E12"):
        run_e12()
