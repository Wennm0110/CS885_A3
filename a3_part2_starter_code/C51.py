from turtle import update
import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10 # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25       # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Suggested constants
ATOMS = 51              # Number of atoms for distributional network
ZRANGE = [0, 200]       # Range for Z projection

# Global variables
EPSILON = STARTING_EPSILON
Z = None                # Online network
z_atoms = None          # Tensor of atom values (Q-value bins)
DELTA_Z = (ZRANGE[1] - ZRANGE[0]) / (ATOMS - 1) # Distance between atoms

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    global Z, z_atoms, DELTA_Z # Use global network and atom variables
    
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    #env.seed(seed)
    test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    #test_env.seed(seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    
    # Define and initialize networks
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    
    # Initialize target network with online network's weights
    Zt.load_state_dict(Z.state_dict())
    
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    
    # Initialize atoms (the discrete Q-value bins)
    z_atoms = torch.linspace(ZRANGE[0], ZRANGE[1], ATOMS).to(DEVICE)
    DELTA_Z = (ZRANGE[1] - ZRANGE[0]) / (ATOMS - 1)
    
    # Return target network and optimizer (Z is global)
    return env, test_env, buf, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, EPSILON_END, STEPS_MAX, Z, z_atoms
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        ## TODO: use Z to compute greedy action
        # START_COMPLETED_CODE
        with torch.no_grad():
            # Get logits from the network
            logits = Z(obs).view(-1, ACT_N, ATOMS)
            # Convert logits to probabilities via softmax
            probs = torch.softmax(logits, dim=2)
            # Calculate expected Q-value for each action
            # Q(s,a) = E[Z(s,a)] = sum(p_i * z_i)
            q_values = (probs * z_atoms).sum(dim=2)
            # Choose the action with the highest expected Q-value
            action = q_values.argmax(dim=1).item()
        # END_COMPLETED_CODE
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# Update networks
def update_networks(epi, buf, Zt, OPT):
    
    global Z, z_atoms, DELTA_Z # Need global Z, z_atoms, and DELTA_Z
    
    loss = 0.
    ## TODO: Implement this function
    # START_COMPLETED_CODE
    
    # Sample a minibatch from the replay buffer
    S, A, R, S_n, D = buf.sample(MINIBATCH_SIZE, t)

    # --- 1. Get next action a* using the online network Z (Double DQN style) ---
    with torch.no_grad():
        logits_Z_sn = Z(S_n).view(-1, ACT_N, ATOMS)
        probs_Z_sn = torch.softmax(logits_Z_sn, dim=2)
        q_values_Z_sn = (probs_Z_sn * z_atoms).sum(dim=2)
        a_star = q_values_Z_sn.argmax(dim=1) # Shape: [B]

    # --- 2. Get next distribution P(s', a*) from the target network Zt ---
    with torch.no_grad():
        logits_Zt_sn = Zt(S_n).view(-1, ACT_N, ATOMS)
        probs_Zt_sn = torch.softmax(logits_Zt_sn, dim=2)
        
        # Select the probability distribution for action a*
        # We need to gather the probs corresponding to a_star
        a_star_idx = a_star.unsqueeze(1).unsqueeze(2).expand(-1, -1, ATOMS)
        next_dist = probs_Zt_sn.gather(1, a_star_idx).squeeze(1) # Shape: [B, ATOMS]

    # --- 3. Project the target distribution (Bellman update) ---
    with torch.no_grad():
        R = R.unsqueeze(1) # Shape: [B, 1]
        D = D.unsqueeze(1) # Shape: [B, 1]
        
        # Calculate the projected atoms: T(z_j) = r + gamma * (1-d) * z_j
        Tz = R + GAMMA * (1 - D) * z_atoms.unsqueeze(0) # Shape: [B, ATOMS]
        
        # Clip the projected values to be within the ZRANGE
        Tz = Tz.clamp(ZRANGE[0], ZRANGE[1])
        
    # --- 4. Distribute the probability (Lillicrap's projection) ---
    # This is the core of C51
    # We project the "next_dist" (probabilities) onto the fixed "z_atoms"
    
    # Calculate indices for lower and upper bounds
    b = (Tz - ZRANGE[0]) / DELTA_Z
    l = b.floor().long()
    u = b.ceil().long()

    # Initialize the target distribution tensor
    m = torch.zeros(MINIBATCH_SIZE, ATOMS).to(DEVICE)
    
    # Distribute probabilities
    # We use scatter_add_ to add probabilities to the correct bins
    # This handles the "linear interpolation" of probabilities
    
    # Handle cases where l == u (atom falls exactly on a bin)
    # These masks prevent double-counting
    l_idx_mask = (l == u)
    u_idx_mask = ~l_idx_mask
    
    # Add to lower bin
    m.scatter_add_(1, l, next_dist * u_idx_mask * (u.float() - b))
    # Add to upper bin
    m.scatter_add_(1, u, next_dist * u_idx_mask * (b - l.float()))
    # Add to bin if l == u
    m.scatter_add_(1, l, next_dist * l_idx_mask)

    # --- 5. Calculate Cross-Entropy Loss ---
    # Get the log-probabilities for the actions taken (A) from the online network Z
    logits_Z_s = Z(S).view(-1, ACT_N, ATOMS)
    log_probs_Z_s = torch.log_softmax(logits_Z_s, dim=2)
    
    # Select the log-probabilities for the action 'A' that was actually taken
    A_idx = A.long().unsqueeze(1).unsqueeze(2).expand(-1, -1, ATOMS)
    log_probs_s_a = log_probs_Z_s.gather(1, A_idx).squeeze(1) # Shape: [B, ATOMS]

    # Calculate the cross-entropy loss between the target distribution 'm' (which is detached)
    # and the predicted log-probabilities 'log_probs_s_a'
    loss = -(m * log_probs_s_a).sum(dim=1).mean()

    # --- 6. Optimization step ---
    OPT.zero_grad()
    loss.backward()
    OPT.step()
    
    loss = loss.item()
    # END_COMPLETED_CODE

    # Update target network
    if epi%TARGET_NETWORK_UPDATE_FREQ==0:
        Zt.load_state_dict(Z.state_dict())

    return loss


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z, z_atoms # Ensure we are using the global variables
    print("Seed=%d" % seed)

    # Create environment, buffer, Zt, optimizer
    # Z, z_atoms are created globally inside this function
    env, test_env, buf, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Zt, OPT) # Z is global, Zt is passed

        # Evaluate for TEST_EPISODES number of episodes
        # Use a separate test policy that is purely greedy
        def test_policy(env, obs):
            global Z, z_atoms
            obs = t.f(obs).view(-1, OBS_N)
            with torch.no_grad():
                logits = Z(obs).view(-1, ACT_N, ATOMS)
                probs = torch.softmax(logits, dim=2)
                q_values = (probs * z_atoms).sum(dim=2)
                action = q_values.argmax(dim=1).item()
            return action

        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, test_policy, render = False)
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
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    plt.show()