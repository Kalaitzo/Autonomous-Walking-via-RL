import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve, keep_same_angles
from robot_environment import RobotEnv

if __name__ == '__main__':
    env = gym.make('Walker2DBulletEnv-v0')  # Create environment
    # env = RobotEnv() # My custom environment for the robot
    # env = gym.make('InvertedPendulumBulletEnv-v0')  # Inverted pendulum environment (for testing)
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])  # Create agent
    n_games = 1000  # Number of games

    filename = 'inverted_pendulum.png'  # Filename
    figure_file = 'plots/' + filename  # Figure file

    best_score = env.reward_range[0]  # Best score
    score_history = []  # Score history
    load_checkpoint = False  # Load checkpoint

    env.render(mode='human')  # Render environment

    if load_checkpoint:
        agent.load_models()  # Load models

    for i in range(n_games):  # Iterate over games
        observation = env.reset()  # Reset environment
        done = False  # Done
        score = 0  # Score

        while not done:  # Iterate while not done
            action = agent.choose_action(observation)  # Choose action
            # TODO: Set the values of the joints that we dont need to move to their initial values (0 or 90 degrees) I
            #  don't remember the correct initial values of the joints. Their indexes are 3, 4, 8, and 9 (not sure)
            # action = keep_same_angles(action, [3, 4, 8, 9])
            observation_, reward, done, info = env.step(action)  # Step
            score += reward  # Update score
            agent.remember(observation, action, reward, observation_, done)  # Remember
            if not load_checkpoint:
                agent.learn()
            observation = observation_  # Observation

        score_history.append(score)  # Append score
        avg_score = np.mean(score_history[-100:])  # Average score

        if avg_score > best_score:  # If average score is greater than best score
            best_score = avg_score  # Update best score
            if not load_checkpoint:
                agent.save_models()

        print('Episode:', i, '| Score: %.1f' % score, '| Average score: %.1f' % avg_score)  # Print results

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)  # Plot results

    env.close()  # Close environment
