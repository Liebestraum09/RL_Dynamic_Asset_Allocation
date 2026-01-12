# RL Asset Allocation: From Daily Trading to Monthly Rebalancing

> **"Is it possible to beat benchmark returns while minimizing negative volatility?"**

## Project Overview
This project implements a Reinforcement Learning (RL) agent for multi-asset portfolio management. It utilizes **PPO, A2C, and SAC** algorithms to optimize asset allocation across five distinct asset classes:
* **SPY** (S&P 500)
* **QQQ** (Nasdaq 100)
* **TLT** (Long-term Treasury)
* **GLD** (Gold)
* **SH** (Inverse S&P 500)

The goal was to build an agent that understands "market regimes"—knowing when to be aggressive and when to be defensive—balancing Alpha (excess returns) with strategic defense.

---

## Performance Analysis

The following results compare the RL agents against a **1/N Benchmark** (holding equal weights of all 5 assets) over the test period (2022-2025).

| Model | Return (%) | Sharpe Ratio | MDD (%) | Strategy Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **PPO** | **28.59%** | **0.84** | **-14.11%** | **Balanced / Tactical.** Best risk-adjusted return. Effectively hedged downside risk using Inverse ETFs during the 2022 bear market. |
| **SAC** | **38.42%** | 0.82 | -17.47% | **Aggressive / Strategic.** Converged to a near-static allocation that maximized exposure to growth assets (QQQ). |
| **A2C** | 23.53% | 0.63 | -16.88% | Baseline performance. Underperformed in terms of stability compared to PPO/SAC. |
| **Benchmark** | 20.47% | 0.66 | -16.78% | Standard 1/N Strategy. |

### Visualization
<img width="1189" height="2490" alt="image" src="https://github.com/user-attachments/assets/9993f0fb-75f0-4e14-a44a-2573d0e4b5be" />

* **PPO:** Demonstrates **Active Management**, increasing `SH` (Inverse) and `GLD` (Gold) exposure during downturns.
* **SAC:** Demonstrates **Passive Optimization**, finding a "Golden Ratio" dominated by equities (`SPY`, `QQQ`) and sticking to it.

---

## Development Journey & Key Insights

This project is a documentation of my engineering journey from a naive trading bot to a asset allocation agent. Below are the 9 key realizations and course corrections made during development.

### 1. Rethinking "Risk": Volatility isn't always bad
* **Initial Intuition:** A good AI should minimize volatility to be safe.
* **The Reality:** Strictly punishing volatility made the agent "cowardly," causing it to hold only cash/bonds and miss the 2023-2024 AI rally.
* **Insight:** **Volatility is the price of admission for high returns.** The strategy was adjusted to manage *downside* volatility (losses) while tolerating *upside* volatility.

### 2. Reward Function Engineering: Balancing Greed and Fear
Designing the objective function was the most critical part of this project.
* **Iteration 1 (Raw Return):** Agent took excessive leverage (High Risk).
* **Iteration 2 (Sharpe Ratio):** Agent became overly conservative (Low Return).
* **Final Design (Greed & Fear):** I implemented a dual-reward system.
    * **Alpha Reward (Greed, Weight 100):** Bonus for outperforming the Benchmark return.
    * **Defense Reward (Fear, Weight 20):** Bonus for having a lower Drawdown than the Benchmark.
    * **Constraint:** Heavy penalty for holding Inverse positions (`SH`) during a Bull Market to prevent betting against the trend.
    * *Result:* Prioritizing Greed over Fear allowed the agent to be aggressive when safe, but the Defense term ensured survival during crashes.

### 3. The Benchmark Trap
* **Initial Approach:** Compared performance against SPY (S&P 500).
* **The Reality:** Since the portfolio includes bonds and gold, comparing a diversified portfolio to 100% stocks was an unfair comparison.
* **Insight:** Switched to a **1/N Benchmark** (Equal Weight). Beating this baseline proves the agent possesses actual asset allocation skills, rather than just riding a bull market.

### 4. Simplicity in Data (Avoiding Look-Ahead Bias)
* **Initial Approach:** Used external APIs (AlphaVantage, FRED) to feed macro indicators like Interest Rates, CPI, and Unemployment.
* **The Reality:** Macro data is reported with a time lag. Using "September CPI" (released in October) for a September decision introduced **Look-Ahead Bias**, invalidating the backtest.
* **Insight:** **"Price Discounts Everything."** I removed external macro data. A drop in TLT (Bond) prices already reflects rising interest rate expectations immediately. Relying solely on Price Action (RSI, MA Divergence) made the model robust and realistic.

### 5. From Micro-Trading to Macro-Allocation
* **Initial Approach:** Built a "Daily Trader" to catch short-term price movements.
* **The Reality:** Daily prices are highly stochastic (noisy). The agent reacted to noise, leading to excessive trading costs and erratic behavior.
* **Insight:** Asset Allocation is about **Macro Trends**, not day trading. Switching to **Monthly Rebalancing** acted as a natural filter, allowing the agent to capture the true market direction.

### 6. The "Overfitting" Sweet Spot
* **Initial Approach:** Planned to train for 1 million timesteps, assuming "more is better."
* **The Reality:** With monthly data (approx. 200 data points), excessive training caused the agent to **memorize** specific historical dates rather than learning general rules.
* **Insight:** Analyzing TensorBoard logs showed the agent learned core logic by 10k steps and stabilized conviction by 50k steps. I stopped training at **60,000 steps** to ensure convergence without overfitting.

### 7. Algorithm Personality: Tactical vs. Strategic
* **Expectation:** All algorithms would converge to a similar strategy.
* **The Reality:**
    * **PPO:** Acted like a **Tactical Trader**, dynamically adjusting weights based on signals.
    * **SAC:** Acted like a **Strategic Optimizer**, finding a static "Golden Ratio" and holding it.
* **Insight:** Different algorithms explore the solution space differently. PPO is better for dynamic hedging, while SAC is better for finding robust long-term static allocations.

### 8. Engineering Constraints: Softmax & Normalization
* **Problem:** Raw price inputs (e.g., $400) and return inputs (e.g., 0.01) had vastly different scales, confusing the neural network gradients.
* **Solution:**
    * **VecNormalize:** Dynamic normalization of inputs to standard deviations.
    * **Softmax Layer:** Enforced Softmax activation on the output layer to mathematically guarantee that the sum of asset weights always equals exactly 1.0 (100%).

### 9. Future Work: Hyperparameter Optimization
* **Limitation:** Due to computational constraints, I focused on validating the strategy logic rather than fine-tuning hyperparameters (Learning Rate, Batch Size, Gamma).
* **Trade-off:** I prioritized **"Winning Now"** (proving the strategy logic works) over "Optimization" (squeezing every basis point). Future updates will find the mathematically optimal parameter set.

---

## Development Workflow & Integrity

This project adopted an **AI-Assisted Engineering** workflow, utilizing **Claude** and **Gemini** for code generation.

* **My Role (The Architect):**
    * Designed the core architecture, logic flow, and state/action space definitions.
    * Identified critical flaws (e.g., look-ahead bias, benchmark mismatch) and directed structural course corrections.
    * Analyzed TensorBoard logs to determine convergence and interpret the "personality" of each algorithm (PPO vs SAC).
* **AI's Role (The Implementer):**
    * Handled Python syntax implementation, boilerplate code for `Stable-Baselines3`, and debugging.
    * Optimized data processing efficiency.

**Why this matters:** Rather than relying on hard-coding, I focused on high-level **problem-solving** and **system design**, demonstrating the ability to effectively orchestrate AI tools for  software engineering.

---

## Methodology & Tech Stack

### Environment Design (Custom Gym Env)
* **Observation Space :** Log Returns, RSI (6-month), MA Divergence (3/12-month).
* **Action Space :** Continuous weights for [SPY, QQQ, TLT, GLD, SH], normalized via Softmax.

### Tools
* **Language:** Python 3.10+
* **RL Library:** Stable-Baselines3, Gymnasium
* **Data:** Pandas, NumPy, Yfinance
* **Visualization:** Matplotlib, TensorBoard
