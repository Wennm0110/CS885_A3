# DRQN 與 C51



## 第一部分：DRQN

### 1.1 簡介 (Introduction)
在本章節中，我們探討了標準 Deep Q-Networks (DQN) 在 **部分可觀測馬可夫決策過程 (POMDPs)** 中的局限性。我們使用了一個修改版的 **部分可觀測 CartPole** 環境，在此環境中代理人 (Agent) 僅能接收不完整的狀態觀測值 (缺失了速度資訊)。為了以解決觀測性不足的問題，我們實作了包含 LSTM 層的 **DRQN**。

### 1.2 DQN Baseline 
我們首先訓練了一個標準 DQN 代理人作為Baseline。該網路僅由全連接層組成，且只接收當前的觀測 frame。

* **觀測問題：** 在缺乏速度資訊的情況下，單一時間點的snapshot是無意義的。代理人無法區分桿子是靜止的還是正在移動的。
* **效能表現：** 如圖 1.1 所示，DQN 代理人無法學出成功的策略。獎勵值停滯在 **40** 左右，顯示代理人無法穩定控制桿子。

![DQN Baseline on POMDP](/a3_part1_starter_code/dqn_results.png)
*圖 1.1: DQN 在部分可觀測 CartPole 上的表現。*

### 1.3 DRQN 實作細節
為了克服部分可觀測性，我們修改了網路結構以包含 recurrent 機制：
* **架構：** `Input -> Linear -> ReLU -> LSTM -> Linear -> Output`。
* **機制：** LSTM 維護一個隱藏狀態 ($h_t, c_t$)，能夠聚合時間序列上的資訊。這使得網路能夠從過往的觀測歷史中，推斷出缺失的速度與動量資訊。
* **訓練：** 我們使用了 Recurrent Replay Buffer 來採樣連續的序列 (Traces)，而非獨立的轉換 (Transitions)。

### 1.4 DRQN 結果與分析
DRQN 代理人的表現如圖 1.2 所示。

![DRQN Performance](/a3_part1_starter_code/drqn_results.png)
*圖 1.2: DRQN (LSTM) 在部分可觀測 CartPole 上的表現。*

**LSTM 層的影響：**
結果清楚地展示了 LSTM 層的有效性。與 DQN 不同，DRQN 代理人成功解決了任務，平均獎勵達到 **150 至 175** 之間 (接近最大值 200)。

**結論：**
標準的前饋網路 (DQN) 在 POMDPs 中失效，是因為觀測到的狀態並非實際的狀態。recurrent 架構允許代理人整合時間資訊，有效地重建控制所需的底層狀態特徵 (如速度)。

---

## 第二部分：類別型 DQN (Categorical DQN, C51)

### 2.1 簡介 (Introduction)
在本章節中，我們實作了 **Categorical (C51)** 分佈式強化學習演算法，以解決隨機環境：**Noisy CartPole** 的問題。此環境在施加的力道上引入了隨機雜訊，並包含了摩擦力，使得轉換動力學具有隨機性，且回報分佈呈現多模態 (Multimodal)。

### 2.2 DQN Baseline
我們在 Noisy CartPole 環境上訓練了標準 DQN 代理人。

* **效能表現：** 如圖 2.1 所示，DQN 雖然學會了解決任務，但表現出顯著的 **不穩定性**。
* **分析：** 在約第 250-300 Episodes 時，效能出現明顯下滑。巨大的變異數 (陰影區域) 表明，在這個充滿雜訊的環境中，僅估計 *期望* 回報 (純量平均值) 不足以進行穩定的學習。

![DQN Baseline on Noisy Env](/a3_part2_starter_code/DQN_results.png)
*圖 2.1: 標準 DQN 在 Noisy CartPole 上的表現。*

### 2.3 C51 實作細節
我們實作了 C51 演算法來模擬完整的價值分佈 $Z(s, a)$。
* **原子 (Atoms)：** 我們使用了 51 個原子，均勻分佈在支撐範圍 `[0, 200]` 之間。
* **投影 (Projection)：** 我們實作了 Distributional Bellman Update，使用線性插值 (Lillicrap's projection) 將目標分佈投影到固定的支撐原子上，並最小化交叉熵損失 (Cross-Entropy Loss)。
* **動作選擇：** 策略根據預測分佈的期望值來選擇動作：$Q(s,a) = \sum p_i z_i$。

### 2.4 C51 結果與比較
C51 代理人的表現如圖 2.2 所示。

![C51 Performance](/a3_part2_starter_code/C51_results.png)
*圖 2.2: C51 在 Noisy CartPole 上的表現。*

**比較 (C51 vs. DQN)：**
1.  **穩定性：** C51 展現了比 DQN 更好的穩定性。當 DQN 在訓練中期遭遇災難性下滑時，C51 保持了更一致的上升趨勢，並且能更平滑地從局部極小值中恢復。
2.  **Robustness：** 透過學習回報的完整分佈而非僅是平均值，C51 捕捉了環境雜訊固有的變異與多模態特性。這提供了更豐富的學習訊號，使代理人能更有效地處理隨機動力學 (雜訊與摩擦)。

### 2.5 結論
結果表明，在隨機環境中，分佈式強化學習 (C51) 相較於純量價值估計 (DQN) 具有實質優勢，提供了改進的穩定性與Robustness。

---

## 使用方式 (Usage)

若要重現實驗結果，請執行對應的 Python 腳本：

**Part 1: DRQN**
```bash
# 執行基準線 (Baseline)
python DQN.py 
# 執行 DRQN 解法
python DRQN.py
```

**Part 2: C51**
```bash
# 執行基準線 (Baseline)
python DQN.py 
# 執行 C51 解法
python C51.py
```
