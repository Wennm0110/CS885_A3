import gym
import numpy as np
# 
# 
# 
# 
# 
# 
# 
# 
import utils2.envs, utils2.seed, utils2.buffers, utils2.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Deep Recurrent Q Learning
# Slide 17
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module4.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
# 
# 
t = utils2.torch.TorchHelper()
DEVICE = t.device
OBS_N = 2               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 2000         # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Global variables
EPSILON = STARTING_EPSILON
Q = None

# Deep recurrent Q network
class DRQN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        ## TODO: Create layers of DRQN (已完成)
        ## START OF CHANGE (USING LSTM) ##
        self.fc1 = torch.nn.Linear(OBS_N, HIDDEN)
        self.relu = torch.nn.ReLU()
        # (1 point) Implement the DRQN model using an LSTM.
        self.lstm = torch.nn.LSTM(HIDDEN, HIDDEN, num_layers=1, batch_first=True)
        self.fc2 = torch.nn.Linear(HIDDEN, ACT_N)
        
        # 用於在 policy 執行期間儲存隱藏狀態 (h, c)
        self.hidden_state = None
        ## END OF CHANGE ##
    
    def forward(self, x, hidden):
        ## TODO: Forward pass (已完成)
        ## START OF CHANGE (USING LSTM) ##
        # x shape: (batch, seq_len, OBS_N)
        # hidden shape: ( (num_layers, batch, HIDDEN), (num_layers, batch, HIDDEN) )
        
        # (batch, seq_len, OBS_N) -> (batch, seq_len, HIDDEN)
        x = self.relu(self.fc1(x))
        
        # x shape: (batch, seq_len, HIDDEN)
        # out shape: (batch, seq_len, HIDDEN)
        # hidden_out shape (tuple): ( (h_n), (c_n) )
        out, hidden_out = self.lstm(x, hidden)
        
        # out shape: (batch, seq_len, HIDDEN) -> (batch, seq_len, ACT_N)
        q_values = self.fc2(out)
        
        return q_values, hidden_out
        ## END OF CHANGE ##

    ## START OF CHANGE (USING LSTM) ##
    def init_hidden(self, batch_size=1):
        # 輔助函數：建立一個初始的隱藏狀態 (h_0, c_0)
        # LSTM 的 hidden state shape: ( (D*num_layers, batch, H_out), (D*num_layers, batch, H_out) )
        h_0 = torch.zeros(1, batch_size, HIDDEN).to(DEVICE)
        c_0 = torch.zeros(1, batch_size, HIDDEN).to(DEVICE)
        return (h_0, c_0)
    ## END OF CHANGE ##

# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def create_everything(seed):
    ## 
    ## 
    utils2.seed.seed(seed)
    env = utils2.envs.TimeLimit(utils2.envs.PartiallyObservableCartPole(), 200)
    #env.seed(seed) # 移除：PartiallyObservableCartPole 沒有 'seed' 屬性
    test_env = utils2.envs.TimeLimit(utils2.envs.PartiallyObservableCartPole(), 200)
    #test_env.seed(seed) # 移除：PartiallyObservableCartPole 沒有 'seed' 屬性
    
    # (1 point) Adjust the replay buffer specification
    # Note: 'recurrent=True' is already set, which is correct for traces
    # 
    # 
    buf = utils2.buffers.ReplayBuffer(BUFSIZE, recurrent=True)
    Q = DRQN().to(DEVICE)
    Qt = DRQN().to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

# Create epsilon-greedy policy
# TODO: Adjust this policy to handle hidden states? (已完成)
def policy(env, obs):
    # (1 point) Adjust the policy function to handle hidden states. (已完成)
    global EPSILON, EPSILON_END, STEPS_MAX, Q
    ## START OF CHANGE (USING LSTM) ##
    # 將 obs 轉換為 (batch=1, seq=1, features) 的形狀以匹配 LSTM 輸入
    obs_tensor = t.f(obs).view(1, 1, OBS_N)

    # env 是 TimeLimit wrapper，_elapsed_steps 在每集(episode)開始時為 0
    # 我們利用這個特性來重置隱藏狀態
    if env._elapsed_steps == 0:
        Q.hidden_state = Q.init_hidden(batch_size=1)
    
    # 處理第一次執行 policy 時 hidden_state 尚未被初始化的情況
    if getattr(Q, 'hidden_state', None) is None:
        Q.hidden_state = Q.init_hidden(batch_size=1)
    
    # 將 hidden_state (一個 tuple) 從上一個計算圖中分離
    Q.hidden_state = (Q.hidden_state[0].detach(), Q.hidden_state[1].detach())


    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
        # 即使是隨機動作，也需要將觀測值傳入網路以更新隱藏狀態
        with torch.no_grad():
            _, Q.hidden_state = Q(obs_tensor, Q.hidden_state)
    else:
        ## TODO: Implement greedy policy (已完成)
        with torch.no_grad():
            # 傳入觀測值和當前的隱藏狀態 (h, c)
            # q_values shape: (1, 1, ACT_N)
            # Q.hidden_state shape: ( (1,1,HIDDEN), (1,1,HIDDEN) )
            q_values, Q.hidden_state = Q(obs_tensor, Q.hidden_state)
        
        # 從 (1, 1, ACT_N) 中取得最佳動作
        action = torch.argmax(q_values).item()
    ## END OF CHANGE ##
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action


# Update networks
def update_networks(epi, buf, Q, Qt, OPT):
    # (1 point) Implement update_networks function in the DRQN.py file. (已完成)
    loss = 0.
    ## TODO: Implement this function (已完成)
    ## START OF CHANGE (USING LSTM) ##
    
    try:
        # 從 recurrent buffer 中採樣一個 minibatch 的 *traces* (s, a, r, s', d)
        S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    except ValueError:
        # 如果緩衝區中的數據不夠（例如，緩衝區大小 < MINIBATCH_SIZE），先不訓練
        return 0.0

    # 獲取實際的 batch size (可能小於 MINIBATCH_SIZE)
    current_batch_size = S.shape[0]

    # 初始化 Q 和 Q_target 的隱藏狀態 (h_0, c_0)
    # h 和 ht 現在都會是 ( (h_0), (c_0) ) 的元組
    h = Q.init_hidden(batch_size=current_batch_size)
    ht = Qt.init_hidden(batch_size=current_batch_size)

    # 1. 計算 Q(s, a)
    # q_values_all shape: (batch, trace_len, ACT_N)
    q_values_all, _ = Q(S, h)
    
    # 使用 .gather() 挑選出實際採取的動作 a 對應的 Q value
    q_values = q_values_all.gather(2, A.long()) # Shape: (batch, trace_len, 1)

    # 2. 計算 Target: y = r + gamma * max_a' Q_target(s', a') * (1-d)
    with torch.no_grad(): # Target 網路不需要計算梯度
        # q2_values_all shape: (batch, trace_len, ACT_N)
        q2_values_all, _ = Qt(S2, ht)
        
        # 找出 max_a' Q_target(s', a')
        q2_values, _ = torch.max(q2_values_all, dim=2, keepdim=True)

        # 計算 y
        targets = R + GAMMA * q2_values * (1.0 - D)

    # 3. 計算 Loss (MSE)
    loss = torch.nn.MSELoss()(q_values, targets)

    # 4. 反向傳播
    OPT.zero_grad()
    loss.backward()
    OPT.step()
    
    loss = loss.item()
    ## END OF CHANGE ##

    # Update target network
    if epi%TARGET_NETWORK_UPDATE_FREQ==0:
        Qt.load_state_dict(Q.state_dict())

    return loss


# Play episodes
# Training function
def train(seed):

    global EPSILON, Q
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        # policy 函數中的隱藏狀態會在 play_episode_rb 內部被重置
        S, A, R = utils2.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Q, Qt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            # policy 函數中的隱藏狀態會在這裡被重置
            S, A, R = utils2.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # (1 point) Produce a graph that shows the performance of the LSTM based DRQN (已完成)
    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'drqn-lstm') # 標籤改為 'drqn-lstm'
    plt.legend(loc='best')
    plt.xlabel('Episodes') # 加上 x 軸標籤
    plt.ylabel('Average Reward') # 加上 y 軸標籤
    plt.title('DRQN (LSTM) Performance on Partially Observable CartPole') # 加上標題
    
    # 儲存圖片而不是顯示
    plt.savefig('drqn_results.png')
    plt.close()
    print("DRQN plot saved to drqn_results.png")