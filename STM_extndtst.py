
"""
Śakti Torus Manifold (STM) © 2026 TS
Coupled‑oscillator model with two geometric symmetry regimes
and full labeled diagnostic output.
"""

import numpy as np
from dataclasses import dataclass
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

@dataclass
class STMGeometry:
    n_up: int = 5
    n_down: int = 4
    mode: str = "offset"   # "offset" or "bindu"

    def reference_angles(self) -> np.ndarray:
        n = self.n_up + self.n_down
        if self.mode == "bindu":
            # fully symmetric, evenly spaced
            return np.linspace(0, TWO_PI, n, endpoint=False)
        # asymmetric, φ‑based quasiperiodic offsets
        step = TWO_PI / (PHI * n)
        return (np.arange(n) * step * PHI) % TWO_PI


# ---------------------------------------------------------------------
# System definition
# ---------------------------------------------------------------------

class STMSystem:
    def __init__(self, geo_mode="offset", record=True):
        self.sizes = (5, 7, 3)
        self.n_total = sum(self.sizes)
        self.geometry = STMGeometry(mode=geo_mode)
        self.alphas = self.geometry.reference_angles()

        # intrinsic frequencies
        self.freq = np.concatenate([
            np.linspace(1.0, 1.3, self.sizes[0]),      # elements
            np.linspace(1.5, 2.1, self.sizes[1]),      # chakras
            np.linspace(0.01, 0.02, self.sizes[2])     # cosmic
        ])

        # coupling matrix (weak cross‑layer links)
        self.K = self._build_coupling()
        self.phases = np.random.uniform(0, TWO_PI, self.n_total)
        self.time = 0.0

        self.record = record
        self.history = defaultdict(list)

    def _build_coupling(self):
        n = self.n_total
        K = np.zeros((n, n))
        idx = np.cumsum((0,) + self.sizes)

        # intra‑layer blocks
        for a in range(3):
            s, e = idx[a], idx[a + 1]
            K[s:e, s:e] = 0.4 * (np.ones((self.sizes[a], self.sizes[a])) - np.eye(self.sizes[a]))

        # inter‑layer links
        K[:5, 5:12] = 0.1;  K[5:12, :5] = 0.1
        K[5:12, 12:] = 0.05; K[:5, 12:] = 0.05
        return K

    # -----------------------------------------------------------------
    # Dynamics
    # -----------------------------------------------------------------

    def step(self, dt):
        dθ = TWO_PI * self.freq.copy()
        for i in range(self.n_total):
            dθ[i] += np.sum(self.K[i] * np.sin(self.phases - self.phases[i]))
        self.phases = (self.phases + dθ * dt) % TWO_PI
        self.time += dt

    # -----------------------------------------------------------------
    # Observables
    # -----------------------------------------------------------------

    def kuramoto_order(self, idx=None):
        if idx is None:
            idx = np.arange(self.n_total)
        return np.abs(np.mean(np.exp(1j * self.phases[idx])))

    def alignment_index(self):
        diffs = np.abs(((self.phases[:, None] - self.alphas[None, :]) + np.pi)
                       % (2 * np.pi) - np.pi)
        return 1.0 - np.mean(np.min(diffs, axis=1)) / np.pi

    def resonance_counts(self, max_order=4, tol=0.05):
        counts = defaultdict(int)
        for i in range(self.n_total):
            for j in range(i + 1, self.n_total):
                ratio = self.freq[i] / self.freq[j]
                for n in range(1, max_order + 1):
                    for m in range(1, max_order + 1):
                        if abs(ratio - n / m) < tol:
                            counts[f"{n}:{m}"] += 1
        return counts

    # -----------------------------------------------------------------
    # Record / run
    # -----------------------------------------------------------------

    def record_state(self):
        if not self.record:
            return
        self.history["time"].append(self.time)
        self.history["coherence_global"].append(self.kuramoto_order())
        self.history["coherence_elements"].append(self.kuramoto_order(slice(0, 5)))
        self.history["coherence_chakras"].append(self.kuramoto_order(slice(5, 12)))
        self.history["coherence_cosmic"].append(self.kuramoto_order(slice(12, 15)))
        self.history["alignment"].append(self.alignment_index())
        self.history["resonances"].append(self.resonance_counts())

    def run(self, t_end=50, dt=0.01, sample_every=10):
        steps = int(t_end / dt)
        for k in range(steps):
            self.step(dt)
            if k % sample_every == 0:
                self.record_state()
        return self


# ---------------------------------------------------------------------
# Reporting / Demo
# ---------------------------------------------------------------------

def summarize(stm, label):
    """Print readable results matching paper structure."""
    print(f"\n───────────────────────────────")
    print(f"{label.upper()} GEOMETRY RESULTS")
    print(f"───────────────────────────────")
    print(f"Bhūmi – Geometric Alignment Stability:")
    print(f"  Final alignment index     = {stm.history['alignment'][-1]:.3f}")

    print(f"\nŚrī – Golden‑Angle Symmetry:")
    print(f"  Rotational variance of phases = {np.var(stm.phases):.3f}")

    print(f"\nTārā – Hierarchical Timescales (subgroup coherences):")
    print(f"  Elements : {stm.history['coherence_elements'][-1]:.3f}")
    print(f"  Chakras  : {stm.history['coherence_chakras'][-1]:.3f}")
    print(f"  Cosmic   : {stm.history['coherence_cosmic'][-1]:.3f}")
    print(f"  Global   : {stm.history['coherence_global'][-1]:.3f}")

    print(f"\nSarasvatī – Resonant Harmonic Attraction:")
    last_res = stm.history["resonances"][-1]
    for ratio, count in sorted(last_res.items()):
        print(f"  Ratio {ratio:<4} → {count:2d} detected pairs")
    print(f"───────────────────────────────\n")


def demo():
    print("ŚAKTI TORUS MANIFOLD SIMULATION\n")

    stm_wake = STMSystem("offset")
    stm_wake.run(t_end=50)
    summarize(stm_wake, "offset‑triangle (waking)")

    stm_sleep = STMSystem("bindu")
    stm_sleep.run(t_end=50)
    summarize(stm_sleep, "bindu‑centered (sleep)")

    print("Simulation complete.\n")


if __name__ == "__main__":
    demo()
