import os
import toml

SWIMLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
"""Path to the extension source directory."""

SWIMLAB_ASSETS_DATA_DIR = os.path.join(SWIMLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

SWIMLAB_ASSETS_METADATA = toml.load(os.path.join(SWIMLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = SWIMLAB_ASSETS_METADATA["package"]["version"]

