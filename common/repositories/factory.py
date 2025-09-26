"""Factory for creating repository instances."""

import logging
from typing import Type, Dict, Any, Optional, Union
from .interfaces import Repository, MarketDataRepository, ContractRepository

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """Factory for creating repository instances with dynamic registration."""
    
    # Registry of repository types mapped to their classes
    _registry: Dict[str, Type[Repository]] = {}
    
    @classmethod
    def register(
        cls, 
        repository_type: str, 
        repository_class: Type[Repository],
        override: bool = False
    ) -> None:
        """Register a repository class.
        
        Args:
            repository_type: Unique identifier for the repository type
            repository_class: Repository class to register
            override: Whether to override existing registrations
            
        Raises:
            ValueError: If repository type already exists and override=False
        """
        if repository_type in cls._registry and not override:
            raise ValueError(
                f"Repository type '{repository_type}' already registered. "
                f"Use override=True to replace."
            )
        
        cls._registry[repository_type] = repository_class
        logger.info(f"Registered repository type: {repository_type} -> {repository_class.__name__}")
    
    @classmethod
    def unregister(cls, repository_type: str) -> bool:
        """Unregister a repository type.
        
        Args:
            repository_type: Repository type to unregister
            
        Returns:
            True if unregistered, False if type was not found
        """
        if repository_type in cls._registry:
            removed_class = cls._registry.pop(repository_type)
            logger.info(f"Unregistered repository type: {repository_type} ({removed_class.__name__})")
            return True
        return False
    
    @classmethod
    def create(
        cls, 
        repository_type: str, 
        data_root: str = "data",
        **kwargs
    ) -> Repository:
        """Create a repository instance.
        
        Args:
            repository_type: Type of repository to create
            data_root: Root directory for data storage
            **kwargs: Additional arguments passed to repository constructor
            
        Returns:
            Repository instance
            
        Raises:
            ValueError: If repository type is not registered
            Exception: If repository creation fails
        """
        if repository_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Unknown repository type: '{repository_type}'. "
                f"Available types: {available_types}"
            )
        
        repository_class = cls._registry[repository_type]
        
        try:
            # Pass data_root as the first positional argument if the repository expects it
            instance = repository_class(data_root, **kwargs)
            logger.debug(f"Created {repository_type} repository with data_root='{data_root}'")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create {repository_type} repository: {e}")
            raise
    
    @classmethod
    def create_with_mixins(
        cls,
        repository_type: str,
        mixins: Optional[list] = None,
        data_root: str = "data",
        **kwargs
    ) -> Repository:
        """Create a repository instance with additional mixins.
        
        Args:
            repository_type: Type of repository to create
            mixins: List of mixin classes to apply
            data_root: Root directory for data storage
            **kwargs: Additional arguments passed to repository constructor
            
        Returns:
            Repository instance with mixins applied
        """
        if repository_type not in cls._registry:
            raise ValueError(f"Unknown repository type: {repository_type}")
        
        base_repository_class = cls._registry[repository_type]
        
        if not mixins:
            # No mixins, use standard creation
            return cls.create(repository_type, data_root, **kwargs)
        
        # Create a dynamic class with mixins
        class_name = f"{base_repository_class.__name__}WithMixins"
        mixin_classes = tuple(mixins) + (base_repository_class,)
        
        try:
            enhanced_class = type(class_name, mixin_classes, {})
            instance = enhanced_class(data_root, **kwargs)
            logger.debug(f"Created {repository_type} repository with mixins: {[m.__name__ for m in mixins]}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create {repository_type} repository with mixins: {e}")
            raise
    
    @classmethod
    def list_types(cls) -> list[str]:
        """List all registered repository types.
        
        Returns:
            List of registered repository type names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_registry(cls) -> Dict[str, Type[Repository]]:
        """Get a copy of the repository registry.
        
        Returns:
            Dictionary mapping repository types to classes
        """
        return cls._registry.copy()
    
    @classmethod
    def is_registered(cls, repository_type: str) -> bool:
        """Check if a repository type is registered.
        
        Args:
            repository_type: Repository type to check
            
        Returns:
            True if registered, False otherwise
        """
        return repository_type in cls._registry
    
    @classmethod
    def get_class(cls, repository_type: str) -> Type[Repository]:
        """Get the repository class for a given type.
        
        Args:
            repository_type: Repository type
            
        Returns:
            Repository class
            
        Raises:
            ValueError: If repository type is not registered
        """
        if repository_type not in cls._registry:
            raise ValueError(f"Unknown repository type: {repository_type}")
        return cls._registry[repository_type]
    
    @classmethod
    def auto_register_from_module(
        cls, 
        module, 
        type_mapping: Optional[Dict[str, str]] = None
    ) -> int:
        """Automatically register repository classes from a module.
        
        Args:
            module: Module containing repository classes
            type_mapping: Optional mapping of class names to repository types
            
        Returns:
            Number of repositories registered
        """
        import inspect
        
        registered_count = 0
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a repository class (has required methods)
            if (hasattr(obj, 'save') and hasattr(obj, 'load') and 
                hasattr(obj, 'exists') and hasattr(obj, 'delete')):
                
                # Determine repository type name
                if type_mapping and name in type_mapping:
                    repo_type = type_mapping[name]
                else:
                    # Convert class name to snake_case repository type
                    repo_type = cls._class_name_to_type(name)
                
                try:
                    cls.register(repo_type, obj)
                    registered_count += 1
                except ValueError as e:
                    logger.warning(f"Could not auto-register {name}: {e}")
        
        logger.info(f"Auto-registered {registered_count} repositories from {module.__name__}")
        return registered_count
    
    @classmethod
    def create_configured(
        cls, 
        config: Dict[str, Any]
    ) -> Dict[str, Repository]:
        """Create multiple repositories from configuration.
        
        Args:
            config: Configuration dictionary with repository specifications
            
        Returns:
            Dictionary mapping repository names to instances
        """
        repositories = {}
        
        for repo_name, repo_config in config.items():
            if not isinstance(repo_config, dict):
                logger.warning(f"Invalid config for repository {repo_name}, skipping")
                continue
            
            repo_type = repo_config.get('type')
            if not repo_type:
                logger.warning(f"No type specified for repository {repo_name}, skipping")
                continue
            
            # Extract configuration parameters
            data_root = repo_config.get('data_root', 'data')
            mixins = repo_config.get('mixins', [])
            kwargs = repo_config.get('kwargs', {})
            
            try:
                if mixins:
                    # Import mixin classes if they are strings
                    mixin_classes = []
                    for mixin in mixins:
                        if isinstance(mixin, str):
                            # Assume mixins are in common.repositories.mixins
                            mixin_module = __import__(
                                f'common.repositories.mixins.{mixin.lower()}',
                                fromlist=[mixin]
                            )
                            mixin_class = getattr(mixin_module, mixin)
                            mixin_classes.append(mixin_class)
                        else:
                            mixin_classes.append(mixin)
                    
                    repo = cls.create_with_mixins(repo_type, mixin_classes, data_root, **kwargs)
                else:
                    repo = cls.create(repo_type, data_root, **kwargs)
                
                repositories[repo_name] = repo
                logger.info(f"Created configured repository: {repo_name} ({repo_type})")
                
            except Exception as e:
                logger.error(f"Failed to create configured repository {repo_name}: {e}")
        
        return repositories
    
    @classmethod
    def _class_name_to_type(cls, class_name: str) -> str:
        """Convert a class name to a repository type name.
        
        Args:
            class_name: Python class name
            
        Returns:
            Repository type name in snake_case
        """
        import re
        
        # Remove 'Repository' suffix if present
        if class_name.endswith('Repository'):
            class_name = class_name[:-10]  # Remove 'Repository'
        
        # Convert CamelCase to snake_case
        snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', class_name).lower()
        
        return snake_case


# Auto-registration helper function
def register_standard_repositories():
    """Register standard repository types.
    
    This function should be called during application startup to register
    the built-in repository types.
    """
    try:
        # Import repository classes
        from .equity_bars import EquityBarRepository
        from .option_bars import OptionBarRepository
        from .option_chains import OptionChainSnapshotRepository
        from .earnings import EarningsRepository
        from .contracts import ContractDescriptionRepository
        from .option_contracts import OptionContractRepository
        
        # Register standard repositories
        RepositoryFactory.register("equity_bars", EquityBarRepository)
        RepositoryFactory.register("option_bars", OptionBarRepository)
        RepositoryFactory.register("option_chains", OptionChainSnapshotRepository)
        RepositoryFactory.register("earnings", EarningsRepository)
        RepositoryFactory.register("contract_descriptions", ContractDescriptionRepository)
        RepositoryFactory.register("option_contracts", OptionContractRepository)
        
        logger.info("Standard repositories registered successfully")
        
    except ImportError as e:
        logger.warning(f"Could not register all standard repositories: {e}")
    except Exception as e:
        logger.error(f"Error registering standard repositories: {e}")


# Convenience factory instances for specific domains
class MarketDataRepositoryFactory:
    """Specialized factory for market data repositories."""
    
    @staticmethod
    def create_equity_bars(
        data_root: str = "data",
        enable_caching: bool = True,
        enable_metrics: bool = True,
        **kwargs
    ) -> Repository:
        """Create an equity bars repository with common configuration."""
        from .mixins import CachingMixin, MetricsMixin
        
        mixins = []
        if enable_caching:
            mixins.append(CachingMixin)
        if enable_metrics:
            mixins.append(MetricsMixin)
        
        return RepositoryFactory.create_with_mixins(
            "equity_bars", mixins, data_root, **kwargs
        )
    
    @staticmethod
    def create_option_chains(
        data_root: str = "data",
        enable_validation: bool = True,
        enable_partitioning: bool = True,
        **kwargs
    ) -> Repository:
        """Create an option chains repository with common configuration."""
        from .mixins import ValidationMixin, PartitioningMixin
        
        mixins = []
        if enable_validation:
            mixins.append(ValidationMixin)
        if enable_partitioning:
            mixins.append(PartitioningMixin)
        
        return RepositoryFactory.create_with_mixins(
            "option_chains", mixins, data_root, **kwargs
        )