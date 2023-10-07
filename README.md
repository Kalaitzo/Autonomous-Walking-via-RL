# Autonomous-Walking-via-RL
This GitHub repository focuses on the development of autonomous walking capabilities for a biped
robot using Reinforcement Learning (RL). It's a part of a research thesis dedicated to exploring
RL-based locomotion control.
## Inverted Pendulum
At first in order to evaluate the performance of the simple actor-critic algorithm and the soft
actor-critic algorithm, the inverted pendulum environment was used.
### Description
![img.png](img/InvPendulum-actor-critic.png)
![img.png](img/InvPendulum-sac.png)
The inverted pendulum is a classic problem in dynamics and control theory.
The goal is to balance a pendulum on a cart that can move along a frictionless track.
### Results
Both the algorithms managed to solve the problem. The simple actor-critic algorithm took around
1000 episodes to solve the problem, while the soft actor-critic algorithm took around 200 episodes thus
proving to be actually more efficient. The visual results for both algorithms can be found in the 
<strong>videos</strong> directory.

## Bipedal Walker
Afterward, the bipedal walker environment was used because the goal is to develop a walking gait
for a real bipedal robot. This environment is more complex than the inverted pendulum one and is a
more realistic representation of the problem.
### Description
![img.png](img/Walker2D-sac.png)
The bipedal walker is a 2D environment where a bipedal robot has to walk on a flat terrain. The
robot has two legs and each leg has two joints. The robot can move forward by applying torque to
the joints. The robot is rewarded for moving forward or standing up and penalized for falling down.
### Results
The only method used to solve the problem was the soft actor-critic algorithm. The algorithm was
able to keep the robot standing. The algorithm didn't manage to make te robot start walking. The
reason for this is that the algorithm didn't go through fine-tuning and perhaps there should be more
episodes in order to solve the problem. Furthermore, this proved that the soft actor-critic algorithm
is very sensitive to the hyperparameters and especially the temperature parameter (trade-off between
exploration and exploitation). Thus, for the next step, the algorithm will be  modified in order to 
automatically tune the temperature parameter. The visual results of the algorithm can be found in the 
<strong>videos</strong> directory.

## Next Steps
The next step is to develop the algorithm with the automatic tuning of the temperature parameter and 
apply it to the bipedal walker environment. Furthermore, all the algorithms will be tested on an 
actual bipedal robot. The robot will be a 10 DOF bipedal robot with 5 DOF per leg with each joint
being a rotational joint. 

