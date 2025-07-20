# ail-mk2

**License Notice:** This repository contains portions derived from [DRAIL](https://github.com/NVlabs/DRAIL?tab=readme-ov-file) under the NVIDIA Source Code License for DRAIL – **non-commercial research & evaluation use only**.

**Environment Setup**
'''
    ./utils/setup.sh
    ./utils/setup_mujoco.sh
    ./utils/expert_data.sh
'''

## Acknowledgements (Anonymous Review Version)

**Code**

- Portions of this code are *derived from* **DRAIL** (NVIDIA Source Code License for DRAIL – non-commercial research/evaluation only).
- Base code adapted from `goal_prox_il`.
- GridWorld environment from `maximecb`.
- Fetch / HandRotate environments customized from OpenAI implementations.
- Ant environment customized by `goal_prox_il`, originating from Farama Foundation.
- SAC implementation from `denisyarats`.
- Maze2D environment based on D4RL.

**Expert Datasets**

- Expert demonstrations for Maze, FetchPick, FetchPush, HandRotate, and AntReach from `goal_prox_il`.

**License / Usage Note**

Original NVIDIA copyright and license retained; this anonymous artifact adds modifications without
introducing any commercial use rights.

