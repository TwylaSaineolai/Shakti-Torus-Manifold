
"""
Śakti Torus Manifold (STM)
A coupled‑oscillator model with two geometric symmetry regimes:
    – offset‑triangle (asymmetric / waking)
    – bindu‑centred (symmetric / sleep)

Author: Twyla Saineolai
"""

import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------

PHI = (1 + np.sqrt(5)) / 2       # golden ratio
TWO_PI = 2 * np.pi


@dataclass
class STMGeometry:
    n_up: int = 5
    n_down: int = 4
    radius: float = 1.0
    mode: str = "offset"          # "offset" or "bindu"

    def reference_angles(self) -> np.ndarray:
        """Return reference angles αₘ depending on geometry."""
        n = self.n_up + self.n_down
        if self.mode == "bindu":
            # evenly spaced, fully symmetric
            alphas = np.linspace(0, TWO_PI, n, endpoint=False)
        else:
            # irregular offsets using golden‑ratio spacing
            step = TWO_PI / (PHI * n)
            alphas = (np.arange(n) * step * PHI) % TWO_PI
        return alphas


# ---------------------------------------------------------------------
# Coupled oscillator system
# ---------------------------------------------------------------------

class STMSystem:
    """Three‑layer coupled‑oscillator system with adjustable symmetry."""

    def __init__(self, geo_mode="offset"):
        # 5+7+3 oscillators
        self.sizes = (5, 7, 3)
        self.n_total = sum(self.sizes)
        self.geometry = STMGeometry(mode=geo_mode)
        self.alphas = self.geometry.reference_angles()

        # Frequencies: scaled across subsystems
        self.freq = np.concatenate([
            np.linspace(1.0, 1.3, self.sizes[0]),             # elements
            np.linspace(1.5, 2.1, self.sizes[1]),             # chakras
            np.linspace(0.01, 0.02, self.sizes[2])            # cosmic
        ])

        # Coupling matrices – weak cross‑layer links
        self.K = self._build_coupling()

        # Initial phases
        self.phases = np.random.uniform(0, TWO_PI, self.n_total)
        self.time = 0.0

    def _build_coupling(self):
        """Block coupling with weak inter‑layer terms."""
        n = self.n_total
        K = np.zeros((n, n))
        sizes = np.cumsum((0,) + self.sizes)
        # intra‑layer uniform
        for a in range(3):
            s, e = sizes[a], sizes[a + 1]
            K[s:e, s:e] = 0.4 * (np.ones((self.sizes[a], self.sizes[a])) - np.eye(self.sizes[a]))
        # inter‑layer weak connections
        K[:5, 5:12] = 0.1
        K[5:12, :5] = 0.1
        K[5:12, 12:] = 0.05
        K[:5, 12:] = 0.05
        return K

    # -----------------------------------------------------------------
    # dynamical evolution
    # -----------------------------------------------------------------
    def step(self, dt: float):
        """Advance phases by one integration step."""
        dθ = TWO_PI * self.freq.copy()
        for i in range(self.n_total):
            dθ[i] += np.sum(self.K[i, :] * np.sin(self.phases - self.phases[i]))
        self.phases = (self.phases + dθ * dt) % TWO_PI
        self.time += dt

    # -----------------------------------------------------------------
    # diagnostic measures
    # -----------------------------------------------------------------
    def alignment_index(self):
        diffs = np.abs(((self.phases[:, None] - self.alphas[None, :]) + np.pi)
             % (2*np.pi) - np.pi)
        mean_dist = np.mean(np.min(diffs, axis=1))
        return 1.0 - mean_dist / np.pi

    def coherence(self):
        """Kuramoto order parameter magnitude."""
        return np.abs(np.mean(np.exp(1j * self.phases)))

    # -----------------------------------------------------------------
    # run
    # -----------------------------------------------------------------
    def run(self, t_end=100, dt=0.01, report=False):
        steps = int(t_end / dt)
        for k in range(steps):
            self.step(dt)
            if report and k % (steps // 10 or 1) == 0:
                print(f"t={self.time:6.2f}, "
                      f"coherence={self.coherence():.3f}, "
                      f"align={self.alignment_index():.3f}")
        return self


# ---------------------------------------------------------------------
# Example execution
# ---------------------------------------------------------------------

def demo():
    print("Śakti Torus Manifold simulation\n")

    # Waking configuration
    stm_wake = STMSystem(geo_mode="offset")
    print("Running offset‑triangle (waking) geometry …")
    stm_wake.run(t_end=50, dt=0.02, report=True)

    # Sleep configuration
    stm_sleep = STMSystem(geo_mode="bindu")
    print("\nRunning bindu‑centred (sleep) geometry …")
    stm_sleep.run(t_end=50, dt=0.02, report=True)

    print("\nSummary:")
    print(f"Waking coherence  : {stm_wake.coherence():.3f}")
    print(f"Sleep coherence   : {stm_sleep.coherence():.3f}")
    print(f"Waking alignments : {stm_wake.alignment_index():.3f}")
    print(f"Sleep alignments  : {stm_sleep.alignment_index():.3f}")


if __name__ == "__main__":
    demo()
