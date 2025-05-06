# SwimmingLab
## Overview
SwimmingLab is a modular framework built on NVIDIA IsaacLab for developing, training, and evaluating drone navigation and control tasks. It provides tools to configure simulation environments, define reinforcement learning tasks, and integrate high-level navigation modules such as semantic mapping. Whether you want to train a drone to follow velocity commands, avoid obstacles, or perform complex waypoint navigation, SwimmingLab offers the building blocks to get started quickly and scale experiments efficiently.

## Structure
The repository, SwimmingLab, consists of `swimlab`, `swimlab_assets`, `swimlab_navigation`, `swimlab_scenes`, and `swimlab_tasks`:
```
.
├── swimlab # Core library extensions (e.g., controllers, utilities)
├── swimlab_assets # Drone-specific assets
├── swimlab_navigation # Navigation (e.g., mapping and planning)
├── swimlab_scenes # Custom scene configurations
└── swimlab_tasks # RL task definitions used for training/evaluation
```

## Installation
To install this repository, follow these steps:

1. Clone the repository.
```
git clone https://github.com/makolon/SwimmingLab.git
```

2. Build docker container.
```
cp .env.sample .env
docker compose build isaac-lab-base
docker compose build isaac-lab-ros2
docker compose build isaac-lab-swim
```

3. Enter `isaac-lab-swim` container.
```
docker compose run isaac-lab-swim
```

> [!NOTE]
> Update the `DISPLAY` environment in the `.env` file using free display. (The display free if it is not in the `/tmp/.X11-unix/` folder of the host machine) Also, change the `WEBPORT` to enable the first free port (Get it by calculating `DISPLAY + 6080`).

```
# If there is no file `/tmp/.X11-unix/X20`
DISPLAY=:20
WEBPORT=6100
```

## Example
You can simulate the `Isaac-Plane-Track-HummingBird-Rotor-v0` environment by running the following command.
```
cd ./scripts/examples/
python env_example.py --task Isaac-Plane-Track-HummingBird-Rotor-v --num_envs 1024
```

## Train and Test Your Policy
Here is an explanation of how to train a policy.

### 1. Execute the Training Script
Once your dataset is ready, follow these steps to train your policy:
```
cd ./scripts/reinforcement_learning/rsl_rl/

python train.py --task Isaac-Plane-Track-HummingBird-Rotor-v0 --num_envs 1024
```

### 2. Test Your Policy
After you have trained a policy, you can evaluate its performance in the simulation environment.
```
cd ./scripts/reinforcement_learning/rsl_rl/

python play.py --task Isaac-Plane-Track-HummingBird-Rotor-v0 --num_envs 1024
```

## Future Extensions
- [ ] 
