import collections
import numpy as np
import random
import torch

# Replay buffer for recurrent networks (stores traces)
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N, recurrent=False):
        self.buf = collections.deque(maxlen = N)
        if not recurrent:
            # This buffer implementation is specifically for recurrent=True
            raise ValueError("ReplayBuffer must be initialized with recurrent=True.")
        self.recurrent = recurrent
    
    # add: add an *entire episode trace*
    # episode_trace: a list of (s, a, r, s2, d) tuples
    def add(self, episode_trace):
        self.buf.append(episode_trace)
    
    # sample: return minibatch of size n (n=batch_size)
    def sample(self, n, t):
        # 1. Sample 'n' episode traces
        #    Ensure we don't sample more than we have
        if len(self.buf) < n:
            raise ValueError(f"Not enough episodes in buffer to sample {n}. Have {len(self.buf)}")
        
        minibatch_episodes = random.sample(self.buf, n)
        
        # 2. Find the max trace length in this batch
        max_len = max(len(epi) for epi in minibatch_episodes)
        
        # 3. Get dimensions from the first transition in the first episode
        #    epi[0_th_episode][0_th_transition][0_th_element (s)]
        obs_dim = len(minibatch_episodes[0][0][0])
        # Actions are discrete, so we'll store them as (B, T, 1)
        
        # 4. Pre-allocate padded tensors
        #    (batch_size, max_len, feature_dim)
        S = torch.zeros((n, max_len, obs_dim))
        A = torch.zeros((n, max_len, 1))
        R = torch.zeros((n, max_len, 1))
        S2 = torch.zeros((n, max_len, obs_dim))
        D = torch.zeros((n, max_len, 1))
        
        # 5. Populate the tensors
        for i, episode in enumerate(minibatch_episodes):
            trace_len = len(episode)
            s_epi, a_epi, r_epi, s2_epi, d_epi = [], [], [], [], []
            
            for transition in episode:
                s, a, r, s2, d = transition
                s_epi.append(s)
                a_epi.append([a])   # Make action shape (1,)
                r_epi.append([r])   # Make reward shape (1,)
                s2_epi.append(s2)
                d_epi.append([d])   # Make done shape (1,)
            
            # Place the trace data into the padded tensor
            S[i, :trace_len, :] = t.f(s_epi)
            A[i, :trace_len, :] = t.l(a_epi) # Use .long() for actions
            R[i, :trace_len, :] = t.f(r_epi)
            S2[i, :trace_len, :] = t.f(s2_epi)
            # Dones should be float for (1.0 - D) calculation
            D[i, :trace_len, :] = t.f(d_epi) 
                
        # 6. Send to device
        return S.to(t.device), A.to(t.device), R.to(t.device), S2.to(t.device), D.to(t.device)