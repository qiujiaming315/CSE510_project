"""Main DQN agent."""
import copy
import torch
import numpy as np

from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.utils import get_hard_target_model_updates, plot_loss, save_checkpoint
from deeprl_hw2.policy import UniformRandomPolicy, LinearDecayGreedyEpsilonPolicy, GreedyPolicy


class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_model:
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    env: the simulation environment.
    check_freq: number of iterations to add a checkpoint (and plot loss value).
    save_path: path to save the checkpoints.
    """

    def __init__(self,
                 q_model,
                 method,
                 preprocessor,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 env,
                 check_freq,
                 save_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.q_model = q_model.to(self.device)
        self.method = method
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.env = env
        self.check_freq = check_freq
        self.save_path = save_path
        self.loss_fn = mean_huber_loss
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=1e-4)
        self.target_network = copy.deepcopy(self.q_model)
        self.target_network.eval()
        self.prepare_policy = UniformRandomPolicy(env.action_space.n)
        self.train_policy = LinearDecayGreedyEpsilonPolicy(1, 0.1, 1000000)
        self.evaluation_policy = GreedyPolicy()
        self.stage = "prepare"
        self.loss_history = []

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        if self.stage == "prepare":
            return self.prepare_policy.select_action()
        else:
            state = self.preprocessor.process_state_for_network(state)
            q_values = self.q_model(state.to(self.device))
            q_values = q_values.cpu().detach().numpy().flatten()
            if self.stage == "train":
                return self.train_policy.select_action(q_values, True)
            else:
                return self.evaluation_policy.select_action(q_values)

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        batch = self.memory.sample(self.batch_size)
        state, action, reward, next_state = self.preprocessor.game_processor.process_batch(batch)
        state = state.to(self.device)
        # Compute the Q-values.
        q_values = self.q_model(state)[torch.arange(self.batch_size), action.to(torch.int64)]
        next_state = next_state.to(self.device)
        # Compute the target values.
        if self.method == "dqn":
            q_target = reward.to(self.device) + self.gamma * torch.amax(self.target_network(next_state), dim=1)
        else:
            target_action = torch.argmax(self.q_model(next_state), dim=1)
            target_q = self.target_network(next_state)[torch.arange(self.batch_size), target_action]
            q_target = reward.to(self.device) + self.gamma * target_q
        loss = self.loss_fn(q_values, q_target)
        self.loss_history.append(loss.cpu().detach().numpy().item())
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.q_model.parameters(), 100)
        self.optimizer.step()

    def fit(self, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        # Fill the replay buffer.
        self.stage = "prepare"
        num_filled = 0
        while num_filled < self.num_burn_in:
            frame = self.env.reset()
            self.preprocessor.reset()
            done = False
            num_step = 0
            while not done and (max_episode_length is None or num_step < max_episode_length):
                frame = self.preprocessor.game_processor.process_state_for_memory(frame)
                action = self.select_action(frame)
                next_frame, reward, done, info = self.env.step(action)
                self.memory.append(frame, action, reward)
                num_filled += 1
                frame = next_frame
                if num_filled == self.num_burn_in:
                    break
            frame = self.preprocessor.game_processor.process_state_for_memory(frame)
            self.memory.end_episode(frame, True)
        # Start training.
        self.stage = "train"
        self.q_model.train()
        num_trained = 0
        while num_trained < num_iterations:
            frame = self.env.reset()
            self.preprocessor.reset()
            done = False
            num_step = 0
            while not done and (max_episode_length is None or num_step < max_episode_length):
                frame = self.preprocessor.game_processor.process_state_for_memory(frame)
                action = self.select_action(frame)
                next_frame, reward, done, info = self.env.step(action)
                self.memory.append(frame, action, reward)
                num_trained += 1
                frame = next_frame
                # Update the main network:
                if num_trained % self.train_freq == 0:
                    self.update_policy()
                # Update the target network.
                if num_trained % self.target_update_freq == 0:
                    get_hard_target_model_updates(self.target_network, self.q_model)
                if num_trained % self.check_freq == 0:
                    plot_loss(self.loss_history, self.save_path)
                    save_checkpoint(self.q_model, self.save_path)
                if num_trained == num_iterations:
                    break
            frame = self.preprocessor.game_processor.process_state_for_memory(frame)
            self.memory.end_episode(frame, True)

    def evaluate(self, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        self.stage = "evaluation"
        self.q_model.eval()
        reward_list = []
        for _ in range(num_episodes):
            frame = self.env.reset()
            self.preprocessor.reset()
            done = False
            total_reward, num_step = 0, 0
            while not done and (max_episode_length is None or num_step < max_episode_length):
                frame = self.preprocessor.game_processor.process_state_for_memory(frame)
                action = self.select_action(frame)
                next_frame, reward, done, info = self.env.step(action)
                total_reward += reward
                num_step += 1
                frame = next_frame
            reward_list.append(total_reward)
        return np.mean(reward_list), np.std(reward_list)
