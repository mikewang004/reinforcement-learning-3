Lunar Lander agent using A2C or Reinforce,
By: Bernard Sohl, Bryce Benz and Mike Wang

The code contains a model which can train on the Lunar Lander-V2 environment using the REINFORCE or A2C algorithm using Pytorch.


Instructions:

1. Download the Assignment3.py file

2. Install the required Python 3 packages:

    Gymnasium, Swig, Gymansium[box2d], Pytorch (torch), Matplotlib, Scipy.Signal

3. Use the following command to train the ACModel on the LunarLander environment, after training it will return a reward-episode plot

   python train.py --render --gamma 0.999 --lr 0.01 --betas 0.9 0.999 --entropy_weight 0.1 --num_episodes 500 --max_steps 10000 --print_interval 10 --method a2c --use_baseline True --N_bootstrap 200

    - render        : A boolean variable indicating whether to render the environment during training. It will render at the max. fps to prevent the render from bottlenecking the training.
    - gamma         : Discount factor for future rewards. It determines the importance of future rewards relative to immediate rewards.
    - lr            : Learning rate for the optimizer. It controls the step size during parameter updates.
    - betas         : Tuple containing the beta parameters for the Adam optimizer. Beta parameters control the exponential moving averages of gradients and squared gradients.
    - entropy_weight: Weight parameter for the entropy term in the loss function. It controls the contribution of the entropy term to the overall loss.
    - num_episodes  : Number of episodes to train the model.
    - max_steps     : Maximum number of steps per episode ().
    - print_interval: Interval for printing the mean reward over a set number of episodes.
    - method        : Algorithm used for training (either 'a2c' or 'reinforce').
    - use_baseline  : Boolean indicating whether to use Baseline Subtraction
    - N_bootstrap   : Number of steps before performing an update (bootstrap size).


4. Save Results : The rewards data will automatically be saved to the Directory Data/Rewards/XXX.csv
                  The pytorch networks will automatically be saved to the Directory Data/Models

