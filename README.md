# Particle Filter Localization

## Overview

This project implements **robot localization using a Particle Filter**.  
The problem addressed here is a localization problem: the environment map is known, but the robot’s position within that map is unknown. The goal is to estimate the robot’s position using sensor measurements and motion updates.

The environment contains several **known and uniquely identifiable landmarks**. The robot can be moved interactively using the mouse. Please refer to the code for details about the simulation setup.

---

## Problem Statement

Given:
- A known 2D map
- A set of known landmarks
- No prior knowledge of the robot’s initial position

Estimate:
- The robot’s position \((x, y)\) using a **Particle Filter**.

---

## Particle Filter Approach

### State Representation

Each particle represents a possible robot position:

\[
X = \{x^0, x^1, \ldots, x^{N-1}\}
\]

where each \(x^i = (x, y)\).

---

## Initialization

Since the robot’s initial position is completely unknown:

- **N particles** are sampled from a **uniform distribution** over the map.
- All particles are assigned equal weights:

\[
W = \{w^0, w^1, \ldots, w^{N-1}\}, \quad w^i = \frac{1}{N}
\]

---

## Main Loop

At each time step, the following steps are executed:

```text
While Z_t:
    Move(X, U_t)
    Update(X, Z_t, W)
    Resample(W)


