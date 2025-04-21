#!/bin/bash

# The following repositories are private, so they need to be added as submodules under the third_party directory.
# To use them as Python modules, you need to install them by executing the provided Shell script.
# This script will handle copying the submodules to the appropriate locations and installing them as Python packages.

# Define source directories
SOURCE_LAB="$SWIMLAB_PATH/source/swimlab"
SOURCE_ASSETS="$SWIMLAB_PATH/source/swimlab_assets"
SOURCE_NAVIGATION="$SWIMLAB_PATH/source/swimlab_navigation"
SOURCE_RL="$SWIMLAB_PATH/source/swimlab_rl"
SOURCE_SCENES="$SWIMLAB_PATH/source/swimlab_scenes"
SOURCE_TASKS="$SWIMLAB_PATH/source/swimlab_tasks"

# Function to link and install a package
install_source() {
    local SOURCE_PATH=$1
    local PACKAGE_NAME=$(basename "$SOURCE_DIR")

    if [ -d "$SOURCE_PATH" ]; then
        # Install the package
        cd "$SOURCE_PATH" || { echo "Failed to enter directory: $SOURCE_PATH"; exit 1; }
        echo "Installing $PACKAGE_NAME with pip"
        ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install -e . || { echo "Failed to install $PACKAGE_NAME"; exit 1; }
        cd - > /dev/null
    else
        echo "Source directory $SOURCE_PATH does not exist. Skipping."
    fi
}

# Install swimlab
install_source "$SOURCE_LAB"

# Install swimlab_assets
install_source "$SOURCE_ASSETS"

# Install swimlab_navigation
install_source "$SOURCE_NAVIGATION"

# Install swimlab_rl
install_source "$SOURCE_RL"

# Install swimlab_scenes
install_source "$SOURCE_SCENES"

# Install swimlab_tasks
install_source "$SOURCE_TASKS"

echo "All tasks completed successfully."

# Execute any additional commands provided to the container
exec "$@"
