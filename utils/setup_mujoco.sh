#!/bin/bash

# Download and extract MuJoCo 2.1
echo "Downloading MuJoCo 2.1..."
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz

# Add environment variables to .bashrc or .zshrc
echo "Setting environment variables..."
if [[ -f ~/.bashrc ]]; then
    echo "export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin" >> ~/.bashrc
elif [[ -f ~/.zshrc ]]; then
    echo "export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210" >> ~/.zshrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin" >> ~/.zshrc
else
    echo "Error: .bashrc or .zshrc file not found"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y libgl1-mesa-dev libosmesa6-dev libglew-dev patchelf

echo "Setup complete!"
echo "✅ Please run 'source ~/.bashrc' (or 'source ~/.zshrc')"