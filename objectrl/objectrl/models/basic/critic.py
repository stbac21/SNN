import copy
import typing
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn as nn

from objectrl.models.basic.ensemble import Ensemble
from objectrl.utils.net_utils import create_loss, create_optimizer

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig

class Critic(nn.Module):
    """
    A critic network that estimates Q-values for state-action pairs.

    Attributes:
        device (torch.device): Device for computations.
        has_target (bool): Flag for presence of target network.
        _tau (float): Polyak averaging factor for target update.
        _gamma (float): Discount factor for rewards.
        model (nn.Module): Main critic network.
        target (nn.Module, optional): Target critic network.
    """

    model: nn.Module
    target: nn.Module | None

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int, shared_trunk: nn.Sequential | None = None) -> None:
        """
        Initialize the critic network.

        Args:
            config: Configuration object containing model parameters
            dim_state: Dimension of the state space
            dim_act: Dimension of the action space
        Returns:
            None
        """
        super().__init__()

        critic = config.model.critic
        self.device = config.system.device
        self.has_target = critic.has_target

        self._tau = config.model.tau
        self._gamma = config.training.gamma

        self.model = critic.arch(
            dim_state,
            dim_act,
            depth=critic.depth,
            width=critic.width,
            act=critic.activation,
            has_norm=critic.norm,
            shared_trunk=shared_trunk if shared_trunk else None # new
        ).to(self.device)

        if self.has_target:
            self.target = critic.arch(
                dim_state,
                dim_act,
                depth=critic.depth,
                width=critic.width,
                act=critic.activation,
                has_norm=critic.norm,
                shared_trunk=shared_trunk if shared_trunk else None # new
            ).to(self.device)
            self.init_target()

    def reduce(self, q_val: torch.Tensor) -> torch.Tensor:
        """
        Reduce Q-values if needed.

        Args:
            q_val (torch.Tensor): Q-values tensor
        Returns:
            torch.Tensor: Reduced Q-values.
        """
        return q_val

    def Q(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values for given state-action pairs.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Q-values for the state-action pairs.
        """
        return self.model(self._prepare_input(state, action))

    @staticmethod
    def _prepare_input(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Concatenate state and action tensors for critic input.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Prepared input tensor for the critic.
        """
        if action.shape == ():
            action = action.view(1, 1)
        return torch.cat((state, action), -1)

    def init_target(self) -> None:
        """
        Initialize target network with weights from the main network.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to initialize"
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters(), strict=True
        ):
            target_param.data.copy_(local_param.data)

    def update_target(self) -> None:
        """
        Update target network parameters using soft update.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to update"
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters(), strict=True
        ):
            # Combine x = (1 - tau) * x + tau * y into a single inplace operation
            target_param.data.lerp_(local_param.data, self._tau)

    def Q_t(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values using the target network.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Target Q-values.
        """
        assert self.has_target, "There is no target network to evaluate"
        return self.target(self._prepare_input(state, action))

    def __getitem__(self) -> "Critic":
        """
        Return self when indexed.

        Args:
            None
        Returns:
            Critic: Self instance.
        """
        return self


class CriticEnsemble(nn.Module, ABC):
    """
    Ensemble of critic networks for robust Q-value estimation.

    Attributes:
        n_members (int): Number of critics in the ensemble.
        config (MainConfig): Configuration object.
        dim_state (int): Dimension of the state space.
        dim_act (int): Dimension of the action space.
        has_target (bool): Flag for target networks.
        _reset (bool): Reset flag from config.
        device (torch.device): Device for computations.
        loss (callable): Loss function.
        _tau (float): Polyak averaging factor.
        _gamma (float): Discount factor.
        model_ensemble (Ensemble[Critic]): Ensemble of critic models.
        optim (torch.optim.Optimizer): Optimizer for ensemble parameters.
        target_ensemble (Ensemble[Critic], optional): Target ensemble.
        iter (int): Training iteration counter.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int, shared_trunk: nn.Sequential | None = None) -> None:
        """
        Initialize the critic ensemble.

        Args:
            config (MainConfig): Configuration object with model parameters.
            dim_state (int): Dimension of the state space.
            dim_act (int): Dimension of the action space.
        Returns:
            None
        """

        super().__init__()

        self.n_members = config.model.critic.n_members
        self.config = config
        self.dim_state = dim_state
        self.dim_act = dim_act
        self.has_target = config.model.critic.has_target
        self._reset = config.model.critic.reset
        self.device = config.system.device
        self.loss = create_loss(config.model, reduction="none")
        self.dim_state = dim_state
        self.dim_act = dim_act
        self._tau = config.model.tau
        self._gamma = config.training.gamma
        
        #SHARE=False # IGNORE - SHOULDN'T BE ACTIVE FOR NOW - ITS FOR SHARED TRUNK IMPLEMENTATION OF ENSEMBLES

        #temp_critic = Critic(config, dim_state, dim_act).model, # new

        self.model_ensemble = Ensemble[nn.Module](
            n_members=int(self.n_members),
            # models are now the first critic + n_members-1 new critics that use the same trunk as the first critic, but with each their own heads (hopefully)
            models=[Critic(config, dim_state, dim_act).model for _ in range(self.n_members)], # previous 
                #[temp_critic[0], Critic(config, dim_state, dim_act, shared_trunk=temp_critic[0].arch.trunk).model] # new
                #if SHARE else
                #...
            device=self.device,
            #sequential=True # THIS IS YIKES, I THINK - TRAINING TIME IS SLOW, AND TAKES INCREASINGLY(!) LONGER TIME AFTER EACH STEP
        )

        self.optim = create_optimizer(self.config.training)(
            self.model_ensemble.parameters()
        )

        if self.has_target:
            #temp_critic = Critic(config, dim_state, dim_act).target, #.model, # new
            self.target_ensemble = Ensemble[nn.Module](
                n_members=int(self.n_members),
                # models are now the first critic + n_members-1 new critics that use the same trunk as the first critic, but with each their own heads (hopefully)
                models=[Critic(config, dim_state, dim_act).target for _ in range(self.n_members)], # previous
                    #[temp_critic[0], Critic(config, dim_state, dim_act, shared_trunk=temp_critic[0].arch.trunk).target] # new
                    #if SHARE else
                    #...
                device=self.device,
                #sequential=True # THIS IS YIKES, I THINK - TRAINING TIME IS SLOW, AND TAKES INCREASINGLY(!) LONGER TIME AFTER EACH STEP
            )
            self.target_ensemble.load_state_dict(self.model_ensemble.state_dict())

        self.iter = 0

    def reset(self) -> None:
        """
        Reset the ensemble models and optimizer.

        Args:
            None
        Returns:
            None
        """
        self.model_ensemble = Ensemble[nn.Module](
            n_members=int(self.n_members),
            models=[
                Critic(self.config, self.dim_state, self.dim_act).model
                for _ in range(self.n_members)
            ],
            device=self.device,
        )

        if self.has_target:
            self.target_ensemble = copy.deepcopy(self.model_ensemble)

        self.optim = create_optimizer(self.config.training)(
            self.model_ensemble.parameters()
        )

    def reduce(self, q_val: torch.Tensor, reduce_type: str) -> torch.Tensor:
        """
        Reduce Q-values from multiple critics according to the configured method.
        Currently supports 'min' or 'mean'. User should add more methods if needed.

        Args:
            q_val (torch.Tensor): Q-values tensor from all critics.
            reduce_type (str): How to reduce the Q-values.
        Returns:
            torch.Tensor: Reduced Q-values.
        """
        if reduce_type == "min":
            return q_val.min(0).values
        elif reduce_type == "mean":
            return q_val.mean(0)
        else:
            raise ValueError(
                f"Unknown reduction method {self.config.model.critic.reduce}. Implement it if needed."
            )

    def _get_single_critic(self, index: int = 0) -> Critic:
        """
        Get a single Critic instance from the ensemble.

        Args:
            index (int): Index of the critic to retrieve.
        Returns:
            Critic: Critic instance.
        """
        single_critic = Critic(self.config, self.dim_state, self.dim_act)

        single_critic.model.load_state_dict(self.model_ensemble[index].state_dict())
        if self.has_target:
            single_critic.target.load_state_dict(  # type: ignore // we know that there is a state_dict
                self.target_ensemble[index].state_dict()
            )
        return single_critic

    def __getitem__(self, index: int) -> Critic:
        """
        Indexing to access a single critic from the ensemble.

        Args:
            index (int): Index of the critic.
        Returns:
            Critic: Critic instance.
        """
        return self._get_single_critic(index)

    def Q(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values for given state-action pairs using all critics.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Q-values from all critics.
        """
        if action is None:
            sa = state
        else:
            sa = torch.cat((state, action), -1)

        return self.model_ensemble(sa)

    @torch.no_grad()
    def Q_t(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values using the target networks.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Target Q-values from all critics.
        """
        assert self.has_target, "There is no target network to evaluate"
        if action is None:
            sa = state
        else:
            sa = torch.cat((state, action), -1)
        return self.target_ensemble(sa)

    def update(
        self, state: torch.Tensor, action: torch.Tensor, y: torch.Tensor
    ) -> None:
        """
        Update critic networks using the provided Bellman targets.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
            y (torch.Tensor): Bellman target values.
        Returns:
            None
        """
        self.optim.zero_grad()
        loss = self.loss(self.Q(state, action), self.model_ensemble.expand(y))
        # Sum over the ensemble members and average over the batches
        loss = loss.sum(0).mean() if self.n_members > 1 else loss.mean()
        loss.backward()
        self.optim.step()
        self.iter += 1

    @torch.no_grad()
    def update_target(self) -> None:
        """
        Update target network parameters using soft update.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to update"
        for key in self.model_ensemble.params.keys():
            m=self.model_ensemble.params[key].data
            t=self.target_ensemble.params[key].data

            t.to(torch.float32).lerp_(
                m.to(torch.float32), self._tau
            )
            # Combine x = (1 - tau) * x + tau * y into a single inplace operation
            #self.target_ensemble.params[key].data.lerp_(
            #    self.model_ensemble.params[key].data, self._tau
            #)

    @abstractmethod
    @torch.no_grad()
    def get_bellman_target(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Calculate the Bellman target for training the critic.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Bellman target tensor.
        """
        pass

# NOTE: W.I.P.
class MHCritic(nn.Module): # MODIFIED COPY OF Critic
    """
    A critic network that estimates Q-values for state-action pairs.

    Attributes:
        device (torch.device): Device for computations.
        has_target (bool): Flag for presence of target network.
        _tau (float): Polyak averaging factor for target update.
        _gamma (float): Discount factor for rewards.
        model (nn.Module): Main critic network.
        target (nn.Module, optional): Target critic network.
    """

    model: nn.Module
    target: nn.Module | None

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int, heads: int=2) -> None:
        """
        Initialize the critic network.

        Args:
            config: Configuration object containing model parameters
            dim_state: Dimension of the state space
            dim_act: Dimension of the action space
        Returns:
            None
        """
        super().__init__()

        critic = config.model.critic
        self.device = config.system.device
        self.has_target = critic.has_target

        self._tau = config.model.tau
        self._gamma = config.training.gamma

        self.model = critic.arch(
            dim_state,
            dim_act,
            depth=critic.depth,
            width=critic.width,
            act=critic.activation,
            has_norm=critic.norm,
        ).to(self.device)

        if self.has_target:
            self.target = critic.arch(
                dim_state,
                dim_act,
                depth=critic.depth,
                width=critic.width,
                act=critic.activation,
                has_norm=critic.norm,
            ).to(self.device)
            self.init_target()

    def reduce(self, q_val: torch.Tensor) -> torch.Tensor:
        """
        Reduce Q-values if needed.

        Args:
            q_val (torch.Tensor): Q-values tensor
        Returns:
            torch.Tensor: Reduced Q-values.
        """
        return q_val

    def Q(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values for given state-action pairs.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Q-values for the state-action pairs.
        """
        return self.model(self._prepare_input(state, action))

    @staticmethod
    def _prepare_input(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Concatenate state and action tensors for critic input.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Prepared input tensor for the critic.
        """
        if action.shape == ():
            action = action.view(1, 1)
        return torch.cat((state, action), -1)

    def init_target(self) -> None:
        """
        Initialize target network with weights from the main network.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to initialize"
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters(), strict=True
        ):
            target_param.data.copy_(local_param.data)

    def update_target(self) -> None:
        """
        Update target network parameters using soft update.

        Args:
            None
        Returns:
            None
        """
        assert self.has_target, "There is no target network to update"
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters(), strict=True
        ):
            # Combine x = (1 - tau) * x + tau * y into a single inplace operation
            target_param.data.lerp_(local_param.data, self._tau)

    def Q_t(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Q-values using the target network.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Target Q-values.
        """
        assert self.has_target, "There is no target network to evaluate"
        return self.target(self._prepare_input(state, action))

    def __getitem__(self) -> "MHCritic":
        """
        Return self when indexed.

        Args:
            None
        Returns:
            Critic: Self instance.
        """
        return self

class MHCriticEnsemble(nn.Module, ABC): # NOTE: MODIFIED COPY OF CriticEnsemble
    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        """
        Initialize the Multiheaded CriticEnsemble.
        - config is Configuration object with model parameters, 
        - dim_state is state space dimension, 
        - dim_act is the action space dimension
        """
        super().__init__()

        self.n_members = config.model.critic.n_members
        self.config = config
        self.dim_state = dim_state
        self.dim_act = dim_act
        self.has_target = config.model.critic.has_target
        self._reset = config.model.critic.reset
        self.device = config.system.device
        self.loss = create_loss(config.model, reduction="none")
        self.dim_state = dim_state
        self.dim_act = dim_act
        self._tau = config.model.tau
        self._gamma = config.training.gamma

        models=MHCritic(config, dim_state, dim_act, heads=int(self.n_members))

        self.model_ensemble = Ensemble[nn.Module](
            n_members=int(self.n_members),
            models=[
                models.models
            ],
            device=self.device,
        )

        self.optim = create_optimizer(self.config.training)(
            self.model_ensemble.parameters()
        )

        if self.has_target:
            self.target_ensemble = Ensemble[nn.Module](
                n_members=int(self.n_members),
                models=[
                    models.targets
                ],
                device=self.device,
            )
            self.target_ensemble.load_state_dict(self.model_ensemble.state_dict())

        self.iter = 0
    
    def reset(self) -> None:
        """
        Reset the critic model, its heads, and the optimizer.
        """
        self.model_ensemble = Ensemble[nn.Module](
            n_members=int(self.n_members),
            models=[
                Critic(self.config, self.dim_state, self.dim_act).model
                for _ in range(self.n_members)
            ],
            device=self.device,
        )

        if self.has_target:
            self.target_ensemble = copy.deepcopy(self.model_ensemble)

        self.optim = create_optimizer(self.config.training)(
            self.model_ensemble.parameters()
        )

    def reduce(self, q_val: torch.Tensor, reduce_type: str) -> torch.Tensor:
        """
        Reduce Q-values from multiple critic heads according to the configured method.
        Currently supports 'min' or 'mean'. User should add more methods if needed.
        - q_val is Q-values tensor from all critic heads, reduce_type is method to be used to reduce the Q-values.
        - Returns the Reduced Q-values (should be either be min() or the Q-values from a randomly chosen critic head, I think).
        """
        if reduce_type == "min":
            return q_val.min(0).values
        elif reduce_type == "mean":
            return q_val.mean(0)
        else:
            raise ValueError(
                f"Unknown reduction method {self.config.model.critic.reduce}. Implement it if needed."
            )

    def _get_single_critic(self, index: int = 0) -> MHCritic:
        """
        Get the critic of the MultiheadCritic instance? idk lol.
            index (int): Index of the critic to retrieve. # FIXME - none of this please
        Returns:
            Critic: Critic instance. # FIXME - i guess? else maybe a MultiheadCritic instance but idk
        """
        single_critic = Critic(self.config, self.dim_state, self.dim_act)

        single_critic.model.load_state_dict(self.model_ensemble[index].state_dict())
        if self.has_target:
            single_critic.target.load_state_dict(  # type: ignore // we know that there is a state_dict
                self.target_ensemble[index].state_dict()
            )
        return single_critic
    
    def __getitem__(self, index: int) -> Critic:
        """
        Indexing to access a single head of the MultiheadCritic.
        - index is Index of the critic head.
        - Returns a Critic(Head?) instance.
        """
        return self._get_single_critic(index) 
    
    def Q(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        "Compute Q-values for given state-action pairs using all critics.
        state is State tensor, action is Action tensor.
        Returns Q-values from all critic heads.
        """
        if action is None:
            sa = state
        else:
            sa = torch.cat((state, action), -1)
        return self.model_ensemble(sa)

    @torch.no_grad()
    def Q_t(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute Q-values using the target networks.
        - state is State tensor, action is Action tensor.
        - Returns Target Q-values from all critic heads.
        """
        assert self.has_target, "There is no target network to evaluate"
        if action is None:
            sa = state
        else:
            sa = torch.cat((state, action), -1)
        return self.target_ensemble(sa)

    # NOTE: W.I.P.
    def update(self, state: torch.Tensor, action: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update critic networks using the provided Bellman targets.
        - state is State tensor, action is Action tensor, y is Bellman target values.
        """
        self.optim.zero_grad()
        loss = self.loss(self.Q(state, action), self.model_ensemble.expand(y))
        # Sum over the ensemble members and average over the batches
        loss = loss.sum(0).mean() if self.n_members > 1 else loss.mean()
        loss.backward()
        self.optim.step()
        self.iter += 1

    @torch.no_grad()
    def update_target(self) -> None:
        """
        Update target network parameters using soft update.
        """
        assert self.has_target, "There is no target network to update"
        for key in self.model_ensemble.params.keys():
            # Combine x = (1 - tau) * x + tau * y into a single inplace operation
            self.target_ensemble.params[key].data.lerp_(
                self.model_ensemble.params[key].data, self._tau
            )
    
    @abstractmethod
    @torch.no_grad()
    def get_bellman_target(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass