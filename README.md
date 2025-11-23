# Reinforcement Learning Assignment: DRQN & C51

This repository contains the implementations and results for Deep Recurrent Q-Learning (DRQN) on Partially Observable CartPole and Categorical DQN (C51) on Noisy CartPole.

## Part 1: Deep Recurrent Q-Learning (DRQN)

### 1.1 Introdu
In this section, we explore the limitations of standard Deep Q-Networks (DQN) in **Partially Observable Markov Decision Processes (POMDPs)**. We utilize a modified **Partially Observable CartPole** environment where the agent receives incomplete state observations (likely missing velocity information). We implement a **DRQN** using an LSTM layer to address these observability issues.

### 1.2 DQN Baseline (The Problem)
We first trained a standard DQN agent as a baseline. The network consists of fully connected layers receiving only the current observation frame.

* **Observation:** Without velocity information, a single snapshot is ambiguous. The agent cannot distinguish between a stationary pole and a moving one.
* **Performance:** As shown in Figure 1.1, the DQN agent fails to learn a successful policy. The reward plateaus around **40**, indicating the agent cannot stabilize the pole.

![DQN Baseline on POMDP](/a3_part1_starter_code/dqn_results.png)
*Figure 1.1: Performance of DQN on Partially Observable CartPole.*

### 1.3 DRQN Implementation
To overcome partial observability, we modified the network to include recurrence:
* **Architecture:** `Input -> Linear -> ReLU -> LSTM -> Linear -> Output`.
* **Mechanism:** The LSTM maintains a hidden state ($h_t, c_t$) that aggregates information over time. This allows the network to implicitly infer the missing velocity and momentum from the history of observations.
* **Training:** We utilized a Recurrent Replay Buffer to sample sequential traces instead of independent transitions.

### 1.4 DRQN Results & Analysis
The performance of the DRQN agent is shown in Figure 1.2.

![DRQN Performance](/a3_part1_starter_code/drqn_results.png)
*Figure 1.2: Performance of DRQN (LSTM) on Partially Observable CartPole.*

**Impact of the LSTM Layer:**
The results clearly demonstrate the effectiveness of the LSTM layer. Unlike the DQN baseline, the DRQN agent successfully learns to solve the task, achieving average rewards between **150 and 175** (approaching the max of 200).

**Conclusion:**
Standard feed-forward networks (DQN) fail in POMDPs because $O_t \neq S_t$. The recurrent architecture allows the agent to integrate temporal information, effectively reconstructing the underlying state features (velocities) required for control.

---

## Part 2: Categorical DQN (C51)

### 2.1 Introduction
In this section, we implement the **Categorical (C51)** distributional RL algorithm to tackle a stochastic environment: **Noisy CartPole**. This environment introduces random noise to the force applied and includes friction, making the transition dynamics stochastic and the return distribution multimodal.

### 2.2 DQN Baseline
We trained a standard DQN agent on the Noisy CartPole environment.

* **Performance:** As seen in Figure 2.1, the DQN agent learns to solve the task but exhibits significant **instability**.
* **Analysis:** Around episodes 250-300, there is a noticeable drop in performance. The high variance (shaded region) suggests that approximating the *expected* return (scalar mean) is insufficient for stable learning in this noisy environment.

![DQN Baseline on Noisy Env](/a3_part2_starter_code/DQN_results.png)
*Figure 2.1: Performance of standard DQN on Noisy CartPole.*

### 2.3 C51 Implementation
We implemented the C51 algorithm to model the full value distribution $Z(s, a)$.
* **Atoms:** We utilized 51 atoms uniformly distributed over the support range `[0, 200]`.
* **Projection:** We implemented the distributional Bellman update, projecting the target distribution onto the fixed support atoms using linear interpolation (Lillicrap's projection) and minimizing the Cross-Entropy loss.
* **Action Selection:** The policy selects actions based on the expected value of the predicted distribution: $Q(s,a) = \sum p_i z_i$.

### 2.4 C51 Results & Comparison
The performance of the C51 agent is shown in Figure 2.2.

![C51 Performance](/a3_part2_starter_code/C51_results.png)
*Figure 2.2: Performance of C51 on Noisy CartPole.*

**Comparison (C51 vs. DQN):**
1.  **Stability:** C51 demonstrates significantly better stability than DQN. While DQN suffered a catastrophic drop mid-training, C51 maintains a more consistent upward trend and recovers more smoothly from local minima.
2.  **Robustness:** By learning the full distribution of returns rather than just the mean, C51 captures the variance and multimodality inherent in the noisy environment. This provides a richer learning signal, allowing the agent to handle the stochastic dynamics (noise and friction) more effectively.

### 2.5 Conclusion
The results suggest that Distributional RL (C51) offers tangible benefits over scalar value estimation (DQN) in stochastic environments, providing improved stability and robustness.

---

## Usage

To reproduce the results, run the corresponding Python scripts:

**Part 1: DRQN**
```bash
# Run baseline
python DQN.py 
# Run DRQN solution
python DRQN.py
```
**Part 2: C51**
```bash
# Run baseline
python DQN.py 
# Run C51 solution
python C51.py
```
