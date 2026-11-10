# SPSGAN — SL(6,C) Spacetime Emergence Simulation

**Geometric–Information Dynamics of Cosmic Evolution under the Six-Dimensional Spacetime Theory**

A CUDA-accelerated simulation that drives the spontaneous emergence of macroscopic (3+1)-dimensional spacetime from the SL(6,C) Lie-algebra moduli space, coupled with a multi-agent cybernetic control system for automated deep-learning decisions injected into the physical evolution engine.

---

## Overview

`space-time-16_0-SPSGAN_CUDA-V23Dual.py` replaces phenomenological cosmological models with a rigorous **Lie-algebra moduli-space evolution**: the microscopic topology of 35 complex generators (70 real degrees of freedom) is mapped to the macroscopic metric spacetime through five consecutive cosmic phase transitions. The evolution is governed by a modified Ricci–Yang–Mills flow with DeTurck gauge fixing, an Eldan–Chen stochastic-localization (Langevin) flow, and an AdS/CFT boundary WZW holographic constraint.

## Core Physics

| Component | Description |
| --- | --- |
| **Moduli space** | SL(6,C) Lie algebra — 35 complex generators, 70 real DOF, built exactly via `build_sl6c_algebra()` |
| **Metric evolution** | Modified Ricci–Yang–Mills flow with DeTurck vector field (strongly parabolic, well-posed) |
| **Stochastic flow** | Eldan–Chen stochastic localization / Langevin SDE with topological potential + level repulsion + Hawking-radiation feedback |
| **Holography** | AdS/CFT boundary WZW loss enforcing bulk–boundary duality self-consistency (Witten 1984) |
| **Phase signature** | (3S+1T) matter / (1S+3T) black hole / (2S+2T) dark energy, with global conservation ΣS = 6, ΣT = 6 |
| **Dark matter ratio** | 75:15 (5:1) algebraic symmetry from broken/unbroken generator dimensions in the Cartan decomposition |
| **Black-hole physics** | Semiclassical Hawking radiation + Page-curve information release |
| **Geometric protection** | Discrete geometric parity (P_G = −1) shielding the shadow sector from radiative decay |

## Five Cosmic Phase Transitions

```
Quantum Foam (UV) ──► Geometric Crystallization ──► (3S+1T) Matter-dominated
  Tr(A) > Θ_crit        Θ_stable < Tr(A) ≤ Θ_crit      Tr(A) ≤ Θ_stable
                                                              │
                                                              ▼
Time-crystal / Dark Energy (IR) ◄────────────── (1S+3T) Black-hole / topological
  w_a < 0, w < −1 (Phantom)                    Hawking radiation + Page curve
```

## Features

- **V23Dual dual-track phase criterion** — `SPSGAN_PHASE_JUDGE=traceA` (original `trace_A` vs `Θ_crit`) or `v22` (three-interval criterion, x = e^{−3N}, D = M − 2ρ_v), aligned with V23 (`6D_Spacetime_Formal_Proof_V23.lean`, 1411 theorems, 0 sorry).
- **Open-parameter response family** (V22 Eq.45, m / M0) with the open-region criterion R0/Λ > 2(m−2)^{3/(m−3)}.
- **Three residual channels** (V22 Eq.57): `residual_RH` / `residual_RE` / `constraint_drift`.
- **Multi-GPU heterogeneous pipeline** — up to 4 GPUs (main engine / planning & topology / reflection & thermodynamics / observation & monitoring).
- **Multi-agent cybernetic control** — Cosmic Orchestrator, Physics Planner (VC-PINN + NTK), Topology Analyst (GUDHI persistent homology), Physics Critic, and P0–P2 engineering cybernetics (PID + rolling window).
- **Manifold steering** — UMAP/PCA-fitted activation-manifold pullback metric replacing white-noise perturbations (with PCA fallback).
- **BFT ghost decoupling** — eigenvector projection removing internal-time ghost modes.
- **Serializable state** — `save_state` / `load_state` persists the V22 criterion slots (fixes resume-criterion loss).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cosmic Orchestrator                        │
│        (periodic scheduler, state scoring, tool registry)     │
└───────┬──────────────┬──────────────┬──────────────┬─────────┘
        │              │              │              │
   [GPU 0]         [GPU 1]        [GPU 2]        [GPU 3]
  Main engine   Planning &      Reflection &   Observation &
  (flow solver,  topology       thermodynamics  monitoring
   metric evo)  (GUDHI TDA,    (critic, VEV,    (SGW Type-I/II,
                Betti)         Langevin)        DM detection)
```

## Installation

### Requirements

- Python 3.11+
- CUDA-enabled GPU (1–4 GPUs recommended; CPU fallback supported)
- PyTorch (CUDA build, `float64` default dtype)
- GUDHI ≥ 3.13 (see note below)
- Optional: `umap-learn`, `pandas`, `scipy`, EvoScientist / AutoResearch for agent control

### GUDHI fix

The `gudhi` build bundled in some conda environments is incomplete (`RipsComplex` import fails / `hera` PermissionError). Point `sys.path` to a standalone `gudhi` 3.13 installation before import; otherwise topology features are silently disabled and Betti numbers remain at the default `[1, 0, 0]`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gudhi umap-learn pandas scipy
```

## Usage

```bash
# Original trace_A criterion (default, backward compatible)
python space-time-16_0-SPSGAN_CUDA-V23Dual.py

# V22 three-interval criterion
SPSGAN_PHASE_JUDGE=v22 python space-time-16_0-SPSGAN_CUDA-V23Dual.py

# Tagged run (distinguish parallel tracks)
SPSGAN_PHASE_JUDGE=v22 SPSGAN_RUN_TAG=runA python space-time-16_0-SPSGAN_CUDA-V23Dual.py
```

## Configuration

Key environment variables and parameters (see the `Config` block in-source for the full set):

| Parameter | Meaning |
| --- | --- |
| `SPSGAN_PHASE_JUDGE` | `traceA` (default) or `v22` — phase-transition criterion |
| `SPSGAN_RUN_TAG` | output filename suffix for dual-track runs |
| `THETA_CRIT` / `THETA_STABLE` | phase-transition critical / stable thresholds |
| `V22_M_RATIO` (m) | response-family exponent (default 6.0) |
| `V22_M0_DUST` (M0) | dust mass offset (default 0.0) |
| `PHASE_SIGNATURE_GAMMA` | geometric phase-signature constraint weight |
| `HAWKING_RAD_THRESHOLD` | Hawking-radiation power threshold for dimension transition |
| `ENABLE_MANIFOLD_STEERING` | toggle UMAP activation-manifold steering |
| `ENABLE_BFT_GHOST_DECOUPLING` | toggle BFT ghost decoupling |

## Project Structure

```
ML-Master/
├── space-time-16_0-SPSGAN_CUDA-V23Dual.py   # main simulation (V23-aligned)
├── space-time-16_0-SPSGAN_CUDA-V22Dual.py   # prior revision (V22 dual-track)
├── AIforScience/
│   ├── EvoScientist/                         # evolutionary strategy agent (optional)
│   └── AutoResearch/                         # research program context (optional)
└── Geometric-Information-Dynamics_Cosmic_Evolution_Mechanism.md   # theory overview
```

## References

- E. Witten, *Non-abelian bosonization in two dimensions*, Commun. Math. Phys. **92**, 455 (1984).
- Six-dimensional spacetime cosmic-evolution panorama and model-algorithm summary (2026-03-30).
- `6D_Spacetime_Phase_Differentiation_Expanded_V23.md` / `6D_Spacetime_Formal_Proof_V23.lean` — V23 alignment reference (1411 theorems, 0 sorry).

## License

This project is research code; use at your own discretion. See the repository license for details.
