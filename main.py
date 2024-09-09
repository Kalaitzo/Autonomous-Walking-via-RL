import pybullet_envs
import gym
from sac_torch import Agent
from utils import *

if __name__ == '__main__':
    env = gym.make('Walker2DBulletEnv-v0')  # Create environment
    # env = RobotEnv() # My custom environment for the robot
    # env = gym.make('InvertedPendulumBulletEnv-v0')  # Inverted pendulum environment (for testing)
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])  # Create agent
    n_games = 10000  # Number of games

    filename = 'Walker2D.png'  # Filename
    figure_file = 'plots/' + filename  # Figure file

    best_score = env.reward_range[0]  # Best score
    score_history = []  # Score history
    load_checkpoint = False  # Load checkpoint
    render = False  # Render
    learn = True  # Learn

    # Initialize an array to store the angles of the joints that are used to achieve the trajectory for each episode
    # 1001 is the number of max steps in each episode, and 6 is the number of joints
    trajectories = np.zeros((n_games, 1001, 6))  # Trajectories of each episode

    if render:
        env.render(mode='human')  # Render environment

    if load_checkpoint:
        agent.load_models()  # Load models

    for i in range(n_games):  # Iterate over games
        observation = env.reset()  # Reset environment
        done = False  # Done
        score = 0  # Score

        # Set the parameters to get the robot's joints angles
        physics_client_id = env.env.physicsClientId  # Physics client ID
        robot_id = env.env.robot.objects[0]  # Robot ID
        joint_ids = [joint.jointIndex for joint in env.env.robot.jdict.values()]  # Joint IDs

        # Get the angles of the joints before starting the episode
        angles_deg = get_angles(robot_id, joint_ids)

        # Initialize the array for the trajectory of the current episode
        trajectory = np.array([])  # Trajectory

        # Append the angles to the trajectory array
        trajectory = np.append(trajectory, angles_deg)

        while not done:  # Iterate while not done
            action = agent.choose_action(observation)  # Choose action
            # TODO: Set the values of the joints that we dont need to move to their initial values (0 or 90 degrees) I
            #  don't remember the correct initial values of the joints. Their indexes are 3, 4, 8, and 9 (not sure)
            # action = keep_same_angles(action, [3, 4, 8, 9])
            observation_, reward, done, info = env.step(action)  # Step
            score += reward  # Update score
            agent.remember(observation, action, reward, observation_, done)  # Remember
            if learn:
                agent.learn()

            # Get the angles of the joints after applying the action/step
            angles_deg = get_angles(robot_id, joint_ids)

            # Append the angles to the trajectory as a new row.
            # TODO: Write down which joint corresponds to each column
            trajectory = np.vstack((trajectory, angles_deg))

            # If the episode is done, store the trajectory
            if done:
                trajectories[i, :trajectory.shape[0], :] = trajectory  # Append the last angles

            observation = observation_  # Observation

        score_history.append(score)  # Append score
        avg_score = np.mean(score_history[-100:])  # Average score over the last 100 games

        if avg_score > best_score:  # If the average score is greater than the best score
            best_score = avg_score  # Set the average score as the best score
            if learn:
                agent.save_models()

        print('Episode:', i, '| Score: %.1f' % score, '| Average score: %.1f' % avg_score)  # Print results

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)  # Plot results

    env.close()  # Close environment
