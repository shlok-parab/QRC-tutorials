# QRC for Timeseries Prediction: Hamiltonians, Encodings, and Hardware Mapping

Quantum Reservoir Computing (QRC) leverages the complex, high-dimensional Hilbert space of a quantum system as a dynamical reservoir. Achieving optimal performance depends heavily on matching the encoding strategy and Hamiltonian to the characteristics of the input signal.

---

## 1. Encoding Strategies & Hamiltonians by Signal Class

| Signal Class | Key Requirement | Recommended Encoding | Recommended Hamiltonian & Dynamics |
| :--- | :--- | :--- | :--- |
| **Spatial / Static** *(e.g., MNIST, Image Classification)* | High dimensionality, complex non-linear separation boundaries. | **Local Parameter Mapping** (e.g., local detunings $\Delta_i$ mapped to spatial features). | **Strongly Interacting / Blockaded Rydberg**: Spacing $d$ close to the blockade radius to maximize quantum entanglement and feature-mixing. |
| **Temporal / Timeseries** *(e.g., Weather, Santa Fe Laser, Mackey-Glass)* | Short-term fading memory (Echo State Property), temporal integration. | **Global Time-Varying Modulation** (e.g., global detuning $\Delta(t)$ or Rabi amplitude $\Omega(t)$ modulated by the signal over time). | **Weakly to Moderately Interacting Rydberg**: Spacing $d$ slightly larger to avoid complete blockade (freezing), allowing the spins to act as a fluid fading-memory buffer. |

### Designing for Timeseries (Weather & Laser Data)
Timeseries forecasting requires the reservoir to balance **fading memory** (retaining past inputs) and **non-linearity** (combining past inputs in non-linear ways).

1. **The Fading Memory Mechanism**:
   When a new value $x(t)$ is encoded as a global detuning $\Delta(t)$ for a duration $t_{\text{seg}}$, the system evolves from its previous state $\psi(t-1)$. The new state $\psi(t)$ is a mixture of the new input and the historical state. 
2. **Critical Tuning Parameters**:
   * **Segment Time ($t_{\text{seg}}$)**: If $t_{\text{seg}}$ is too long, the system relaxes to its ground/steady state for the current input, completely erasing past memory. If $t_{\text{seg}}$ is too short, the spins do not have time to interact, reducing the non-linearity. Usually, $t_{\text{seg}} \in [0.1, 0.5]$ µs is optimal.
   * **Atom Spacing ($d$)**: Controls the coupling strength $V_{ij} = C_6/d^6$. If $d$ is too small, the system enters the Rydberg blockade regime, freezing spin flips and destroying memory. If $d$ is too large, the atoms behave as independent qubits (no feature-mixing).

---

## 2. Hardware Platform Mapping

Predicting timeseries data on real hardware requires mapping the abstract dynamics to the native physical capabilities of the platform.

### A. QuEra Aquila (Analog Neutral-Atom Simulator)
Aquila is a native analog simulator based on Rydberg atoms, making it the **ideal hardware platform** for analog QRC.
* **How it Maps**: 
  * The atoms are arranged in a 1D or 2D array.
  * The timeseries sequence is programmed as a time-varying global detuning pulse shape $\Delta(t)$, where each step of the timeseries corresponds to a flat segment of the pulse.
  * Readout is performed by mapping the Z-component of each spin (excited state vs. ground state) at the end of each segment.
* **Pros**: Native execution of continuous Rydberg Hamiltonian, zero Trotterization error, excellent scale (up to 256 qubits).

### B. IBM Eagle (Gate-Based Superconducting Processor)
Superconducting processors like Eagle cannot natively run continuous analog Hamiltonians. Instead, they require **Digitized QRC (DQRC)**.
* **How it Maps**:
  * The time evolution operator $U = e^{-iHt}$ is Trotterized into a sequence of discrete quantum gates.
  * A typical DQRC step consists of:
    1. Projecting the timeseries value $x(t)$ onto single-qubit rotation gates: $R_z(\theta)$ where $\theta \propto x(t)$.
    2. Applying a fixed entangling layer (e.g., CNOT or CZ gates on a heavy-hex lattice) to mix the states.
  * The measurement step is performed by reading out expectation values (e.g., $\langle \sigma_z^i \rangle$) after each step.
* **Pros**: High gate speeds, local single-qubit control.
* **Cons**: Noise and coherence limits restrict the Trotter steps.

---

## 3. Mathematical Optimization: Diagonal Expectation Shortcut

In simulated QRC (like the Python code in `QRC Demo MNIST.ipynb`), calculating the expectation values is the second-largest bottleneck after the matrix exponential.

### The Standard Approach
The current code performs a vector-matrix-vector product for every tracking operator $O$ at every time step:
$$\langle O \rangle = \langle\psi| O |\psi\rangle = \text{cp.vdot}(\psi, O \psi)$$
Since we track 8 single-atom $Z$ operators and 28 pairwise $ZZ$ operators, this requires **36 matrix-vector multiplications** of size $256 \times 256$ per step.

### The Diagonal Shortcut
Because the Pauli $Z$ and $ZZ$ operators are built using Kronecker products of diagonal matrices, **every tracking operator $O$ is diagonal**. 
Let $d_O$ be the 1D diagonal vector of $O$ (size $256$). The expectation value simplifies to:
$$\langle O \rangle = \sum_k |\psi_k|^2 O_{kk} = \vec{p} \cdot \vec{d}_O$$
where $\vec{p} = |\psi|^2$ is the real-valued probability density vector of the statevector.

We can precompute the diagonals of all $Z$ and $ZZ$ operators once globally, and then compute all 36 expectation values instantly using a single matrix-vector multiplication, completely avoiding loops:

```python
# ── PRECOMPUTATION (Run Once) ──────────────────────────────────────────
# Extract the diagonals of all operators and stack them
Z_diags = cp.array([cp.diag(Zi) for Zi in Z_ops])     # Shape: (8, 256)
ZZ_diags = cp.array([cp.diag(ZiZj) for ZiZj in ZZ_ops]) # Shape: (28, 256)

# ── IN THE LOOP (Run Per Step) ──────────────────────────────────────────
# 1. Compute probability density vector (1D, real-valued, size 256)
prob = cp.abs(psi)**2

# 2. Compute all expectation values instantly via matrix-vector multiplication
out_z_step = Z_diags @ prob     # Shape: (8,)
out_zz_step = ZZ_diags @ prob   # Shape: (28,)
```

### Complexity Reduction
* **Standard Vector Dot**: $36 \times O(D^2)$ complex operations.
* **Diagonal Shortcut**: $1 \times O(D)$ probability density + $2 \times O(N_{\text{ops}} \cdot D)$ real operations.
* **Speedup**: $\sim \mathbf{250\times}$ reduction in operations for the measurement loop.
