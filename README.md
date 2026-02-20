# SNN
temp repo for current version of SNN+DRL for EvoGym with ObjectRL

CriticEnsemble with sequential=True seems required to use the snntorch.Leaky() layers inbetween the fully connected torch.linear layers in an nn.Sequential wrapper, as the leaky layers must be initialized with hidden states (necessary to work with nn.Sequential), which makes the the neuron states be created as network buffers instead of returning them as torch tensors along with the spike tensor output (binary vector). Thus, only the spikes are returned and fed through to the next layer of the nn.Sequential model (as intended). Membrane potentials of the neurons in the snn.leaky layers (aka. "states") are only important during the forward call in the MLP, i think.

Problem:
When using sequential=True on creation of CriticEnsembles, the training time increases as steps increase. It is not linearly nor constantly increasing with every single step, but the average time over 100 steps in the environment takes increasingly longer and longer.

02-20 11:05:03 INFO     Episode:    1	N-steps:     499	Reward:      0.023 (first 500 steps are warmup)

02-20 11:05:58 INFO     Episode:    2	N-steps:     999	Reward:     -0.020 (55 secs)

02-20 11:07:22 INFO     Episode:    3	N-steps:    1499	Reward:      0.057 (84 secs)

02-20 11:09:21 INFO     Episode:    4	N-steps:    1999	Reward:      0.021 (119 secs)

02-20 11:11:55 INFO     Episode:    5	N-steps:    2499	Reward:      0.035 (154 secs)

02-20 11:15:01 INFO     Episode:    6	N-steps:    2999	Reward:      0.044 (186 secs)

02-20 11:18:41 INFO     Episode:    7	N-steps:    3499	Reward:      0.048 (220 secs)


As seen on these logs (run config found in objectrl/_logs/evogym-walker/td3/<seed>/<datetime>), every 500 steps take longer and longer durations. this pattern does not occur with leaky layers ONLY in the actor-network. I cannot get leaky layers to work with the critic networks in the CriticEnsemble of TD3, so I do not know how this affects training time (other than certainly making it take ~5x longer overall).
The samme pattern of increasingly longer training times apply when running on GPU.
