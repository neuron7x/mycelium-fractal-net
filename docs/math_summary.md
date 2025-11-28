# Mathematical Foundations of MyceliumFractalNet

This document provides a mathematical summary of the key concepts implemented in MyceliumFractalNet v4.1.

## Table of Contents

1. [Fractal Layer Architecture](#fractal-layer-architecture)
2. [Turing Pattern Growth](#turing-pattern-growth)
3. [Nernst Potential Activation](#nernst-potential-activation)
4. [STDP Learning Rule](#stdp-learning-rule)
5. [Network Topology](#network-topology)

---

## Fractal Layer Architecture

### Self-Similar Structure

The fractal layer architecture organizes neurons in a self-similar hierarchy with depth *d*. At each depth level *k*, the layer processes information at a different scale:

$$
y = \sum_{k=1}^{d} w_k \cdot f_k(x)
$$

Where:
- *d* is the fractal depth
- *w_k* is the learnable weight for pathway *k*
- *f_k(x)* is the transformation at depth *k*

### Pathway Combination

Pathways are combined using softmax-weighted averaging:

$$
w_k = \frac{\exp(\alpha_k)}{\sum_{j=1}^{d} \exp(\alpha_j)}
$$

This ensures the pathway weights form a probability distribution.

---

## Turing Pattern Growth

### Reaction-Diffusion System

Network topology evolves according to the Gray-Scott reaction-diffusion model:

$$
\frac{\partial A}{\partial t} = D_A \nabla^2 A - AI^2 + f(1 - A)
$$

$$
\frac{\partial I}{\partial t} = D_I \nabla^2 I + AI^2 - (k + f)I
$$

Where:
- *A* is the activator concentration
- *I* is the inhibitor concentration
- *D_A*, *D_I* are diffusion coefficients
- *f* is the feed rate
- *k* is the kill rate

### Connection Mask Generation

Connections are formed where activator exceeds a threshold *θ*:

$$
M_{ij} = \mathbb{1}[A_{ij} > \theta]
$$

This creates spatially-organized, self-organizing connection patterns.

---

## Nernst Potential Activation

### Biological Foundation

The Nernst equation describes the equilibrium potential of an ion across a membrane:

$$
E = \frac{RT}{zF} \ln\left(\frac{[X]_{out}}{[X]_{in}}\right)
$$

Where:
- *R* = 8.314 J/(mol·K) is the gas constant
- *T* is temperature in Kelvin
- *z* is the ion valence
- *F* = 96,485 C/mol is Faraday's constant

### Neural Network Adaptation

For computational efficiency, we use a scaled tanh activation:

$$
\sigma(x) = A \cdot \tanh\left(\gamma \cdot (x - \theta) \cdot \frac{RT}{F}\right)
$$

Where:
- *A* is the amplitude (default: 1.0)
- *γ* is the scaling factor (default: 10.0)
- *θ* is the threshold potential (default: -0.065V)
- *T* = 310K (body temperature)

### Properties

- Bounded output: *σ(x) ∈ [-A, A]*
- Smooth gradient for stable training
- Biologically interpretable parameters

---

## STDP Learning Rule

### Spike-Timing Dependent Plasticity

STDP modifies synaptic weights based on the relative timing of pre- and post-synaptic spikes:

$$
\Delta w = 
\begin{cases}
A^+ \exp\left(-\frac{\Delta t}{\tau^+}\right) & \text{if } \Delta t > 0 \\
-A^- \exp\left(\frac{\Delta t}{\tau^-}\right) & \text{if } \Delta t < 0
\end{cases}
$$

Where:
- *Δt = t_post - t_pre* is the time difference
- *A+*, *A-* are learning rates
- *τ+*, *τ-* are time constants

### Trace-Based Implementation

For efficient computation, we use exponentially decaying traces:

$$
x_i(t) = x_i(t-1) \cdot e^{-\Delta t/\tau^+} + S_i^{pre}(t)
$$

$$
y_j(t) = y_j(t-1) \cdot e^{-\Delta t/\tau^-} + S_j^{post}(t)
$$

Weight updates are then:

$$
\Delta w_{ij} = A^+ \cdot S_j^{post}(t) \cdot x_i(t) - A^- \cdot y_j(t) \cdot S_i^{pre}(t)
$$

### Soft Bounds

To prevent unbounded weight growth:

$$
\Delta w_{ij} \leftarrow \Delta w_{ij} \cdot (w_{max} - w_{ij}) \cdot w_{ij}
$$

---

## Network Topology

### Graph Representation

The network is represented as a directed graph *G = (V, E)* where:
- *V* is the set of neurons (nodes)
- *E* is the set of connections (edges)

### Connectivity Metrics

**Average path length:**
$$
L = \frac{1}{n(n-1)} \sum_{i \neq j} d(i, j)
$$

**Clustering coefficient:**
$$
C_i = \frac{2 |\{e_{jk}\}|}{k_i(k_i - 1)}
$$

Where *k_i* is the degree of node *i*.

### Fractal Dimension

The network exhibits self-similarity characterized by the fractal dimension:

$$
D_f = \lim_{r \to 0} \frac{\log N(r)}{\log(1/r)}
$$

Where *N(r)* is the number of boxes of size *r* needed to cover the network.

---

## References

1. Turing, A. M. (1952). "The Chemical Basis of Morphogenesis." *Philosophical Transactions of the Royal Society B*.

2. Bi, G. Q., & Poo, M. M. (1998). "Synaptic modifications in cultured hippocampal neurons." *The Journal of Neuroscience*.

3. Nernst, W. (1889). "Die elektromotorische Wirksamkeit der Jonen." *Zeitschrift für Physikalische Chemie*.

4. Gray, P., & Scott, S. K. (1984). "Autocatalytic reactions in the isothermal, continuous stirred tank reactor." *Chemical Engineering Science*.
