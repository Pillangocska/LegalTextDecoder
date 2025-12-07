import os
import yaml
from typing import Dict, Any

# Priority order for config file location:
config_paths = [
    '/opt/airflow/dags/repo/config/config.yaml',
    'config.yaml',
    'config/config.yaml',
    '..config/config.yaml',
    '../config/config.yaml',
    '../../config/config.yaml',
    '../../../config/config.yaml',
    '/config/config.yaml',
    '../config/config.yaml'
]

class ConfigManager:
    """Centralized configuration manager"""

    _instance: 'ConfigManager | None' = None
    _config: 'Dict[str, Any] | None' = None

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file"""

        config_file = None
        attempted_paths = []

        for path in config_paths:
            if path:  # Only check non-None paths
                attempted_paths.append(path)
                if os.path.exists(path):
                    config_file = path
                    break

        if not config_file:
            raise FileNotFoundError(
                f"FATAL: Configuration file not found. No fallback defaults will be used. "
                f"Attempted locations: {attempted_paths}. "
                f"Please ensure the config file exists or set AIRFLOW_CONFIG_PATH environment variable."
            )

        try:
            with open(config_file, 'r', encoding="UTF-8") as stream:
                self._config = yaml.safe_load(stream)

            if not self._config:
                raise ValueError(f"Configuration file {config_file} is empty or invalid")

            # Store the config file path for reference
            self._config_file_path = config_file
            print(f"Successfully loaded configuration from: {config_file}")

        except yaml.YAMLError as e:
            raise ValueError(f"FATAL: Invalid YAML format in config file {config_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to load config from {config_file}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'database.host')"""
        # Assert that config is loaded
        assert self._config is not None, "Configuration not loaded"

        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        # Assert that config is loaded
        assert self._config is not None, "Configuration not loaded"

        if section not in self._config:
            raise KeyError(f"FATAL: Configuration section '{section}' not found in config file. Available sections: {list(self._config.keys())}")
        return self._config[section]

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get_section('database')

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get_section('logging')

    def get_cas_config(self) -> Dict[str, Any]:
        """Get CAS configuration"""
        return self.get_section('cas')

    @property
    def config_file_path(self) -> str:
        """Get the path of the loaded config file"""
        return getattr(self, '_config_file_path', 'Unknown')


# Singleton
config = ConfigManager()

"""
USAGE
from src.config.config_manager import config
# using the convinience function
db_config = config.get_database_config()
db = PostgresManager(**db_config)
# using simple dot notation
schema_name = config.get('misc.schema_name')
user_id = config.get('misc.user_id')
"""
