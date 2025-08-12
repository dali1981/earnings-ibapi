"""
Configuration for data lineage tracking system.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from .core import OperationType


@dataclass
class LineageConfig:
    """Configuration for lineage tracking system."""
    
    # Storage configuration
    base_path: str = "./lineage_data"
    enable_persistence: bool = True
    
    # Tracking configuration
    track_repository_operations: bool = True
    track_dataframe_operations: bool = True
    track_file_operations: bool = False
    
    # Operation tracking filters
    enabled_operation_types: List[OperationType] = field(default_factory=lambda: [
        OperationType.READ,
        OperationType.WRITE,
        OperationType.TRANSFORM,
        OperationType.AGGREGATE
    ])
    
    # Performance configuration
    max_parameter_size: int = 1000  # Max size of parameters to store
    max_metadata_size: int = 5000   # Max size of metadata to store
    cleanup_after_hours: Optional[float] = None  # Auto-cleanup old data
    
    # Repository-specific settings
    repository_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Visualization settings
    max_graph_nodes: int = 1000     # Max nodes in visualization
    max_graph_depth: int = 10       # Max depth for lineage traversal
    
    @classmethod
    def from_env(cls) -> 'LineageConfig':
        """Create configuration from environment variables."""
        return cls(
            base_path=os.getenv('LINEAGE_BASE_PATH', './lineage_data'),
            enable_persistence=os.getenv('LINEAGE_ENABLE_PERSISTENCE', 'true').lower() == 'true',
            track_repository_operations=os.getenv('LINEAGE_TRACK_REPOS', 'true').lower() == 'true',
            track_dataframe_operations=os.getenv('LINEAGE_TRACK_DATAFRAMES', 'true').lower() == 'true',
            track_file_operations=os.getenv('LINEAGE_TRACK_FILES', 'false').lower() == 'true',
            max_parameter_size=int(os.getenv('LINEAGE_MAX_PARAM_SIZE', '1000')),
            max_metadata_size=int(os.getenv('LINEAGE_MAX_METADATA_SIZE', '5000')),
            max_graph_nodes=int(os.getenv('LINEAGE_MAX_GRAPH_NODES', '1000')),
            max_graph_depth=int(os.getenv('LINEAGE_MAX_GRAPH_DEPTH', '10')),
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'LineageConfig':
        """Load configuration from JSON/YAML file."""
        import json
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                # Try YAML if available
                try:
                    import yaml
                    config_data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML configuration files")
        
        # Convert operation types from strings
        if 'enabled_operation_types' in config_data:
            config_data['enabled_operation_types'] = [
                OperationType(op_type) if isinstance(op_type, str) else op_type
                for op_type in config_data['enabled_operation_types']
            ]
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if key == 'enabled_operation_types':
                # Convert operation types to strings
                config_dict[key] = [op.value for op in value]
            else:
                config_dict[key] = value
        
        return config_dict
    
    def save_to_file(self, config_path: Path, format: str = 'json') -> None:
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if format.lower() == 'json':
                import json
                json.dump(config_dict, f, indent=2)
            elif format.lower() == 'yaml':
                try:
                    import yaml
                    yaml.dump(config_dict, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML required for YAML configuration files")
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def get_repository_setting(self, repo_name: str, setting_name: str, default: Any = None) -> Any:
        """Get repository-specific setting."""
        repo_settings = self.repository_settings.get(repo_name, {})
        return repo_settings.get(setting_name, default)
    
    def set_repository_setting(self, repo_name: str, setting_name: str, value: Any) -> None:
        """Set repository-specific setting."""
        if repo_name not in self.repository_settings:
            self.repository_settings[repo_name] = {}
        self.repository_settings[repo_name][setting_name] = value
    
    def should_track_operation(self, operation_type: OperationType) -> bool:
        """Check if operation type should be tracked."""
        return operation_type in self.enabled_operation_types
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate paths
        if not self.base_path:
            issues.append("base_path cannot be empty")
        
        # Validate numeric settings
        if self.max_parameter_size <= 0:
            issues.append("max_parameter_size must be positive")
        
        if self.max_metadata_size <= 0:
            issues.append("max_metadata_size must be positive")
        
        if self.max_graph_nodes <= 0:
            issues.append("max_graph_nodes must be positive")
        
        if self.max_graph_depth <= 0:
            issues.append("max_graph_depth must be positive")
        
        # Validate operation types
        if not self.enabled_operation_types:
            issues.append("enabled_operation_types cannot be empty")
        
        return issues


# Default configuration instance
DEFAULT_CONFIG = LineageConfig()


def get_config() -> LineageConfig:
    """Get the current lineage configuration."""
    # Try environment variables first
    try:
        return LineageConfig.from_env()
    except Exception:
        # Fall back to default configuration
        return DEFAULT_CONFIG


def create_sample_config_file(config_path: Path) -> None:
    """Create a sample configuration file."""
    sample_config = LineageConfig(
        base_path="./data/lineage",
        enable_persistence=True,
        track_repository_operations=True,
        track_dataframe_operations=True,
        track_file_operations=False,
        enabled_operation_types=[
            OperationType.READ,
            OperationType.WRITE,
            OperationType.TRANSFORM
        ],
        max_parameter_size=2000,
        max_metadata_size=10000,
        cleanup_after_hours=168.0,  # 1 week
        repository_settings={
            "equity_bars": {
                "track_all_operations": True,
                "enable_volume_tracking": True
            },
            "option_bars": {
                "track_all_operations": True,
                "track_greeks": True
            },
            "option_chains": {
                "track_snapshot_operations": True,
                "track_pricing_changes": False
            }
        },
        max_graph_nodes=2000,
        max_graph_depth=15
    )
    
    sample_config.save_to_file(config_path)