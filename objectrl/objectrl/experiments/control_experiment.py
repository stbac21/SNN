# -----------------------------------------------------------------------------------
# ObjectRL: An Object-Oriented Reinforcement Learning Codebase
# Copyright (C) 2025 ADIN Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------------

import snntorch as snn

import time

import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm

from objectrl.config.config import MainConfig
from objectrl.experiments.base_experiment import Experiment
from objectrl.utils.utils import tonumpy, totorch
    
class ICM(nn.Module):
    """
    Intrinsic Curiosity Module - the implementation (somewhat) follows https://pathak22.github.io/noreward-rl/resources/icml17.pdf
    - Takes "st", "at", and "st1": the current state, the action taken the the agent, and the next state resulting from taking said action, respectively.
    - Returns "r_i" at time t: the intrinsic reward signal/bonus that drives curiosity to encourage explorative actions that result in the acquisition of novel information.
    - Encodes states into feature space representations that are used in the forward and inverse dymamics models.
    - "r_i" is calculated w.r.t. eq. (6), where η > 0 is a scaling factor for the intrinsic reward bonus: η/2*L2norm(φˆ(st+1)−φ(st+1))^2
    """
    class FDM(nn.Module):
        """
        Foward Dymanics Model - φˆ(st+1)=f(φ(st),at;θF) - eq. (4).
        - Takes "φ(st)" and "at": a feature-space-state-representation of the current state and the action taken by the agent, respectively.
        - Returns "φˆ(st+1)": an approximated prediction of the feature state representation of the next state - φ(st+1).
        - Model parameters "θF" are optimized to minimize the following loss function: LF(φ(st),φˆ(st+1)) = 1/2*L2norm(φˆ(st+1)−φ(st+1))^2 - eq (5)
        """
        def __init__(self,  dim_in: int, dim_out: int, width: int):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(dim_in, width), nn.ReLU(inplace=True), nn.Linear(width, dim_out))
        
        def forward(self, at, Øst): # at is the action taken by the agent, Øst is the feature-space-state-representation of the current state.
            return self.model(torch.cat([at, Øst], dim=-1)) # concatenate the action and the feature-space-state-representation vector and feed it through the net.

    class IDM(nn.Module):
        """
         Inverse Dymanics Model - ât=g(st,st+1;θI) - eq. (2).
        - Takes "φ(st)" and "φ(st+1)": the feature-space-state-representations of the current and next state, respectively.
        - Returns "ât": an approximated prediction of the action taken by the agent to get from st to st+1.
        - Model parameters "θI" are optimized to minimize a loss function that measures the discrepancy between ât and at: LI(ât,at) - eq. (3).
        """
        def __init__(self, dim_in: int, dim_out: int, width: int):
            super().__init__()
            # this implementation differs in not using an initializer - if one is to be used, add the following:
            #   nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            #   nn.init.zeros_(self.fc1(...))
            self.model = nn.Sequential(nn.Linear(dim_in, width), nn.ReLU(inplace=True), nn.Linear(width, dim_out), nn.Tanh())

        def forward(self, Øst, Øst1): # Øst is the feature-space-state-representation of the current state, Øst1 is the feature-space-state-representation of the next state.
            return self.model(torch.cat([Øst, Øst1], dim=-1)) # concatenate the two feature-space-state-representation vectors and feed it through the net.

    class Encoder(nn.Module):
        """
        takes s_t and s_t+1 and produces φ(s_t) and φ(s_t+1) - the encoded feature-space-state-representations of the current and next state.
        """
        def __init__(self, dim_in: int, width: int, dim_out: int):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(dim_in, width), nn.ReLU(), nn.Linear(width, dim_out), nn.ReLU())

        def forward(self, x):
            return self.model(x)

    def __init__(self, obs_dim: int, act_dim: int, width: int=256, feature_dim:int=512):
        super().__init__()
        # State encoder
        self.encoder = ICM.Encoder(dim_in=obs_dim, width=width, dim_out=feature_dim)    # state in, feature-space-state-representation out, intrinsic reward bonus returned.

        # IDM head
        self.idm = ICM.IDM(dim_in=feature_dim+feature_dim, width=width, dim_out=act_dim)    # two feature-space-state-representations in, action out.

        # FDM head
        self.fdm = ICM.FDM(dim_in=act_dim+feature_dim, width=width*4, dim_out=feature_dim)    # action + feature-space-state-representation in, feature-space-state-representation out.

        # Optimizers
        self.inv_optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.idm.parameters()), lr=3e-4)
        self.fwd_optimizer = torch.optim.Adam(self.fdm.parameters(), lr=1e-4)
    
    # perform per-step updates and return intrinsic reward bonus as R_i = η/2*L2norm(φˆ(st+1)−φ(st+1))^2
    def step(self, at, st, st1):
        # encode s_t and s_t+1
        Øst = self.encoder(st)      # encoded current state φ(s_t)
        Øst1 = self.encoder(st1)    # encoded next state φ(s_t+1)
        
        # predict action and calculate inverse loss
        at_hat = self.idm(Øst, Øst1)    # encoded current state + encoded next state --> predicted approximation of the action taken by the agent.
        inv_loss = torch.nn.functional.mse_loss(at_hat, at, reduction='mean')    # dicrepancy between predicted action and actual action

        # predict next state and calculate forward loss
        Øst1_hat = self.fdm(at, Øst.detach())    # action + encoded current state --> φˆ(st+1); predicted approximation of the encoding of next state
        fwd_loss = (0.5*torch.norm(Øst1_hat - Øst1.detach(), p=2, dim=-1)**2).mean()    # equation (5) + .mean()

        # update inverse path
        self.inv_optimizer.zero_grad()
        inv_loss.backward()
        self.inv_optimizer.step()

        # update forward path
        self.fwd_optimizer.zero_grad()
        fwd_loss.backward()
        self.fwd_optimizer.step()

        # N is a positive constant - intrinsic reward bonus scaling factor (use range 0.001-0.1). NOTE - SHOULD BE TUNED
        # walker_N=0.01, Climber_N=0.001, upstepper=TBD, thrower=TBD, platformjumper=TBD
        
        #N=0.01 # Walker - ICM: width=256, feature_space=512
        # Agent with ICM benefits from higher N, which im guessing is due to the simplicity of the environment.
        # Agent seems to learn much faster on evo-walker with ICM enabled, but it was trained with slope=2 instead of 25,
        # which might also have had a significant positive effect (though it'd matter more if network was 
        # deeper; as it was only 2 deep, the effect might've bene negligible and thus the ICM has had a big positive effect)
        
        #N=0.001 # Climber - ICM: width=256, feature_space=512
        # Agent with ICM didnt work with N=.01 on evo-climber. with ICM and N=.001 the agent works far better than TD3 without ICM.

        #N=0.001 # seems too low - ICM: width=256, feature_space=512
        #N=0.005 # seems promising, but didnt work (, probably) due to too small replay buffer of 200k - ICM: width=256, feature_space=512
        N=0.008 # WORKS - but might be too high - i think 5e-3 is better - ICM: width=512, feature_space=1024 - WORKS AT ~950k STEPS!!!
        #N=0.005 # RUN THIS NEXT - try this again with ICM: width=512, feature_space=1024
        # Upstepper: N=0.008 works, but takes quite long. N=0.005 might be better; TBD.
        

        # FIXME - R_i should probably be normalized in some way but idk - it seems to be working anyway so maybe its fine as-is.
        R_i = N*fwd_loss#/512 # feature_dim - ??? FIXME

        #print("\t\tINTRINSIC REWARD SIGNAL: ", R_i)
        return R_i.detach() # needed if gradients leak, but idk if they do. - FIXME

class ControlExperiment(Experiment):
    """
    The Experiment class for training and evaluating.
    This class defines the core training loop, manages interaction with the environment,
    performs evaluations at regular intervals, and handles model saving and logging.

    Args:
    max_steps : int
        Maximum number of training steps.
    warmup_steps : int
        Number of initial steps using random actions before policy-based action selection.
    device : torch.device
        Device (CPU/GPU) used for tensor computations.
    """

    def __init__(self, config: "MainConfig"):
        super().__init__(config)

        # Retrieve training parameters from the configuration
        self.max_steps: int = self.config.training.max_steps
        self.warmup_steps: int = self.config.training.warmup_steps
        self.device = torch.device(config.system.device)
        self._vectorized_eval = self.config.training.parallelize_eval
        self.verbose = self.config.verbose

    def train(self) -> None:
        """
        Runs the training loop for the agent, managing interactions with the environment,
        learning updates, evaluations, and logging.

        Args:
            None
        Returns:
            None
        """
        time_start = time.time()

        # Variables for State at time t, action at time t, and state at time t+1 
        # These are needed in the ICM (Intrinsic Curiosity Module) to provide intrisic reward bonus
        ICM_st, ICM_at, ICM_st1 = None, None, None
        
        IR, ER = 0, 0

        icm = ICM(self.env.observation_space._shape[0], self.env.action_space._shape[0], width=512, feature_dim=1024).to(device=self.device)

        # Dictionary to store rewards and steps for logging
        information_dict = {
            "episode_rewards": torch.zeros(self.max_steps),
            "episode_steps": torch.zeros(self.max_steps),
            "step_rewards": np.empty((2 * self.max_steps), dtype=object),
        }

        # Initialize the environment and state
        state, _ = self.env.reset()
        state = totorch(state, device=self.device)
        r_cum = np.zeros(1)
        episode = 0
        e_step = 0
        
        # Training loop
        print("TRAINING...")
        t_=time.time()
        for step in tqdm(
            range(self.max_steps),
            leave=True,
            disable=not self.config.progress,
        ):
            # progress print (DELETE)
            if step % 100 == 0 and step != 0:
                t__ = time.time()
                print(f"STEP: {step} - {t__-t_} seconds")
                t_=time.time()
            e_step += 1

            # Reset agent periodically if configured
            if (
                step > self.warmup_steps
                and self.config.training.reset_frequency > 0
                and step % self.config.training.reset_frequency == 0
            ):
                print(f"RESETTING... (STEP: {step}) - CURRENTLY DISABLED AND DOES NOTHING")
                #self.agent.reset()
                #snn.utils.reset(self.agent.actor)

            # Evaluate the agent at specified intervals
            if (
                step > self.warmup_steps
                and step % self.config.training.eval_frequency == 0
            ):
                IR, ER = 0, 0
                t0 = time.time()
                #print(f"\t--> {t0-t_} SECONDS!\nEVALUATING... (STEP: {step})")
                self.eval(step)
                t1 = time.time()
                print(f"EVAL AT STEP {step} DONE IN {t1-t0} SECONDS")
                #print(f"\t--> {t1-t0} SECONDS!\nTRAINING...")

            # Select an action (random during warmup, policy-based afterward)
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
                action = totorch(np.clip(action, -1.0, 1.0), device=self.device)
                act_dict = {"action": action}
            else:
                act_dict = self.agent.select_action(state)
                action = act_dict["action"].clip(-1.0, 1.0)


            # Take a step in the environment
            next_state, reward, terminated, truncated, info = self.env.step(
                int(action) if self._discrete_action_space else tonumpy(action)
            )
            next_state = totorch(next_state, device=self.device)

            R_e = reward
            
            ICM_ENABLED = False
            if ICM_ENABLED:
                ICM_at, ICM_st, ICM_st1 = action, state, next_state
                R_i = icm.step(ICM_at, ICM_st, ICM_st1)
                
                # alternatively, scale R_i in a way that makes it a viable multiplier, as EvoGym constantly provides a ~non-zero extrinsic reward (though often a small one)
                reward = R_e + R_i

                # running sum for statistics
                ER += R_e
                IR += R_i

                if step % 500 == 0:
                    print(f"\t R_i + R_e = {IR} + {ER} = {IR+ER}  ~  {IR/(IR+abs(ER))*100}%")
                    ER, IR = 0, 0

                #reward = R_e * R_i # ALT.? use if R_i is scaled properly for this to make sense

            transition_kwargs = {
                **act_dict,
                "state": state,
                "next_state": next_state,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "step": step + 1,
            }
            transition = self.agent.generate_transition(**transition_kwargs)

            # Store the transition in replay buffer
            self.agent.store_transition(transition)

            # Log per-step reward
            information_dict["step_rewards"][self.n_total_steps + step] = (
                episode,
                step,
                #reward,
                R_e, # log only the extrinsic reward signal - maybe also add the intrinsic one separately? idk
            )

            state = next_state  # Update state
            #r_cum += reward  # Update cumulative reward
            r_cum += R_e  # Update cumulative extrinsic reward signals

            # Perform learning updates at specified intervals
            if (
                step >= self.warmup_steps
                and (step % self.config.training.learn_frequency) == 0
            ):
                #t0 = time.time()
                #print(f"\t--> {t0-t_} SECONDS!\nUPDATING AGENT... (STEP: {step})")
                self.agent.learn(
                    max_iter=self.config.training.max_iter,
                    n_epochs=self.config.training.n_epochs,
                )
                #t_ = t1 = time.time()
                #print(f"\t--> {t1-t0} SECONDS!\nTRAINING...")

            # Episode termination
            if terminated or truncated:
                information_dict["episode_rewards"][episode] = r_cum.item()
                information_dict["episode_steps"][episode] = step

                # Save episode summary
                self.agent.logger.episode_summary(episode, step, information_dict)

                # Reset the environment for the next episode
                state, _ = self.env.reset()
                state = totorch(state, device=self.device)
                r_cum = np.zeros(1)
                episode += 1
                e_step = 0

                # RESET MEMBRANE POTENTIALS OF SNNs - ONLY USE THIS WITH T=1
                #snn.utils.reset(self.agent.actor)
                #snn.utils.reset(self.agent.critic) # ONLY USE THIS WITH SPIKING CRITIC
                #print("RESETTING ENV")

            # Save model and logs at specified intervals
            if step % self.config.logging.save_frequency == 0:
                self.agent.logger.save(information_dict, episode, step)
                self.agent.save()

        # Final evaluation after training
        self.eval(step)
        time_end = time.time()
        self.agent.save()
        self.agent.logger.save(information_dict, episode, step)
        self.agent.logger.log(f"Training time: {time_end - time_start:.2f} seconds")

    # ruff: noqa: C901
    @torch.inference_mode()
    def eval(self, n_step: int) -> None:
        """
        Evaluates the agent over multiple episodes in the evaluation environment.

        Args:
            n_step (int): The current training step at which evaluation is performed.
        Returns:
            None
        """
        self.agent.eval()  # Set agent to evaluation mode

        # Save RNG states
        torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        # Set deterministic seed for eval
        eval_seed = self.config.system.seed + 12345
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)

        # Store rewards for evaluation episodes
        results = torch.zeros(self.config.training.eval_episodes)

        if self._vectorized_eval:
            states, _ = self.eval_env.reset()
            states = totorch(states, device=self.device)

            dones = torch.zeros(self.config.training.eval_episodes, dtype=torch.bool)
            while not torch.all(dones):
                actions = self.agent.select_action(states, is_training=False)["action"]

                if self._discrete_action_space:
                    actions = actions.int()

                next_states, rewards, term, trunc, _ = self.eval_env.step(
                    tonumpy(actions)
                )

                done = torch.tensor(term) | torch.tensor(trunc)
                results += torch.tensor(rewards) * (
                    ~done
                )  # only add reward to running environments
                dones |= done

                states = totorch(next_states, device=self.device)

        else:
            # Run multiple evaluation episodes
            for episode in range(self.config.training.eval_episodes):
                state, info = self.eval_env.reset()
                state = totorch(state, device=self.device)

                done = False

                frames = []
                while not done:
                    # Select action using the agent's policy (without exploration)
                    action = self.agent.select_action(state, is_training=False)[
                        "action"
                    ]

                    frame = self.eval_env.render()
                    frames.append(frame)  # save it

                    # Execute action in the environment
                    next_state, reward, term, trunc, info = self.eval_env.step(
                        int(action) if self._discrete_action_space else tonumpy(action)
                    )

                    # Check termination condition
                    done = term or trunc
                    # Update state and record reward
                    state = totorch(next_state, device=self.device)
                    results[episode] += reward

                import imageio
                imageio.mimsave(f"objectrl/_logs/videos/eval_{n_step}_{episode}.mp4", frames, fps=30, macro_block_size = None)

                # If using Sparse MetaWorld env, adjust reward to reflect success
                # mean_reward in save_eval_results will be equal to success rate
                if "success" in info and self.config.env.sparse_rewards:
                    results[episode] = reward + 1.0

        self.agent.logger.save_eval_results(n_step, results)
        if self.verbose:
            tqdm.write(f"{n_step}: {results.mean():.4f} +/- {results.std():.4f}")
        self.agent.train()

        # Restore RNG states
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
