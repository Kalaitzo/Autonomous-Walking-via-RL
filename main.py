import pybullet_envs
import pybullet as p
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

    filename = 'Walker2D.png'  # Filename
    figure_file = 'plots/' + filename  # Figure file

    best_score = env.reward_range[0]  # Best score
    score_history = []  # Score history
    load_checkpoint = True  # Load checkpoint
    render = True  # Render
    learn = True  # Learn

    # Set an array to store the angles of the joints that are used to achieve the trajectory for each episode
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

        # Initialize the array to be store in the 3D array with the trajectories of each episode
        trajectory = np.array([])  # Trajectory

        # TODO: Write a function in the utils.py file to get the angles of the joints of the robot
        #  that has inputs the robot id, client id, and the joint ids. The function should return the angles of the
        #  joints in degrees. The function should be called get_joint_angles(robot_id, physics_client_id, joint_ids)
        angles = []  # Angles
        # Get the angles of the joints before applying any action
        for joint_id in joint_ids:
            angle = p.getJointState(robot_id, joint_id)[0]  # Get joint angle
            angles.append(angle)  # Append angle
        # TODO:==========================================

        angles = np.array(angles)  # Convert to a numpy array
        angles_deg = np.rad2deg(angles)  # Convert to degrees

        # Append the angles to the trajectory
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
            # TODO: Update this part with the function that will be created in the utils.py file
            angles = []  # Array for the new angles
            for joint_id in joint_ids:
                angle = p.getJointState(robot_id, joint_id)[0]  # Get a joint angle (This is in radians probably)
                angles.append(angle)

            angles = np.array(angles)  # Convert to a numpy array
            angles_deg = np.rad2deg(angles)  # Convert to degrees

            # Append the angles to the trajectory as a new row.
            # TODO: Write down which joint corresponds to each column
            trajectory = np.vstack((trajectory, angles_deg))

            # If the episode is done, append the trajectory (array of steps) of that episode to the 3D array
            # At the end of the episode each column will have the angles of all actions
            # taken for each joint
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
