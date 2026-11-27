import numpy as np

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [SX, SY, SZ]


def bloch_to_rho(r):
    r = np.asarray(r, dtype=float)
    return 0.5 * (I2 + r[0] * SX + r[1] * SY + r[2] * SZ)


def rho_to_bloch(rho):
    return np.array([np.real(np.trace(rho @ p)) for p in PAULI])


def random_bloch_axis(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def random_bloch_state(rng):
    return random_bloch_axis(rng)


def haar_unitary(dim, rng):
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    q *= d / np.abs(d)
    return q


def random_pure_rho(n, rng):
    psi = haar_unitary(2 ** n, rng)[:, 0]
    return np.outer(psi, psi.conj())


def embed(op1, k, n):
    # Qubit 0 is the leftmost tensor factor.
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, op1 if q == k else I2)
    return out


def projector_from_axis(b):
    b = np.asarray(b, dtype=float)
    return 0.5 * (I2 + b[0] * SX + b[1] * SY + b[2] * SZ)


def yes_prob(rho, proj_full):
    return float(np.real(np.trace(proj_full @ rho)))


def apply_unitary(rho, u):
    return u @ rho @ u.conj().T


def dephase_qubit(rho, n, k, axis, p):
    if p <= 0.0:
        return rho
    proj = embed(projector_from_axis(axis), k, n)
    comp = np.eye(2 ** n, dtype=complex) - proj
    dephased = proj @ rho @ proj + comp @ rho @ comp
    if p >= 1.0:
        return dephased
    return (1.0 - p) * rho + p * dephased


def partial_trace_qubit0(rho, n):
    reshaped = rho.reshape([2, 2 ** (n - 1), 2, 2 ** (n - 1)])
    return np.trace(reshaped, axis1=1, axis2=3)


def selective_partial_collapse(rho, n, k, axis, p, outcome):
    if p <= 0.0:
        return rho
    proj = embed(projector_from_axis(axis), k, n)
    pa = np.real(np.trace(proj @ rho))
    if outcome:
        collapsed = proj @ rho @ proj / pa
    else:
        comp = np.eye(2 ** n, dtype=complex) - proj
        collapsed = comp @ rho @ comp / (1.0 - pa)
    return (1.0 - p) * rho + p * collapsed


def asymmetric_partial_collapse(rho, n, k, axis, p_up, p_down, outcome):
    if outcome:
        return selective_partial_collapse(rho, n, k, axis, p_up, 1)
    return selective_partial_collapse(rho, n, k, axis, p_down, 0)


def replicability_1q(rho, n, k, axis, p):
    proj = embed(projector_from_axis(axis), k, n)
    pa = float(np.real(np.trace(proj @ rho)))
    return (1.0 - p) * (pa * pa + (1.0 - pa) ** 2) + p


def random_pure_rho_entangled(n, rng, mode="haar"):
    if mode == "haar":
        return random_pure_rho(n, rng)
    if mode == "product":
        psi = np.array([[1.0 + 0j]])
        for _ in range(n):
            psi = np.kron(psi, haar_unitary(2, rng)[:, 0])
        return np.outer(psi, psi.conj())
    if mode == "maxpair":
        pair = haar_unitary(4, rng)[:, 0]
        rest = np.array([[1.0 + 0j]])
        for _ in range(n - 2):
            rest = np.kron(rest, haar_unitary(2, rng)[:, 0])
        psi = np.kron(pair, rest)
        return np.outer(psi, psi.conj())
    raise ValueError(f"Unknown init mode: {mode}")


def entanglement_entropy_qubit0(rho, n):
    r0 = partial_trace_qubit0(rho, n)
    evals = np.linalg.eigvalsh(r0)
    evals = np.clip(evals.real, 0.0, 1.0)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log(evals)))


def is_valid_rho(rho, tol=1e-9):
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if abs(np.real(np.trace(rho)) - 1.0) > tol:
        return False
    return float(np.linalg.eigvalsh(rho).min()) > -tol


def luders_qubit_table(r, a, b):
    r, a, b = map(np.asarray, (r, a, b))
    pA1 = 0.5 * (1.0 + a @ r)
    pB1 = 0.5 * (1.0 + b @ r)
    r_after_B = (b @ r) * b
    r_after_A = (a @ r) * a
    pA2 = 0.5 * (1.0 + a @ r_after_B)
    pB2 = 0.5 * (1.0 + b @ r_after_A)
    return dict(pA1=float(pA1), pB1=float(pB1), pA2=float(pA2), pB2=float(pB2))


def luders_qubit_analytics(r, a, b):
    r, a, b = map(np.asarray, (r, a, b))
    qA = 0.5 * ((a @ r) - (a @ b) * (b @ r))
    qB = 0.5 * ((b @ r) - (a @ b) * (a @ r))
    return dict(q_A=float(qA), q_B=float(qB), R=float(qA - qB), S=float(qA + qB))
