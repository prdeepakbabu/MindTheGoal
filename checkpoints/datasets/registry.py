"""Dataset registry for managing available loaders."""

from typing import Dict, Type, Optional
from datasets.base_loader import BaseDatasetLoader
from datasets.multiwoz_loader import MultiWOZLoader
from datasets.sgd_loader import SGDLoader
from datasets.custom_loader import CustomLoader


class DatasetRegistry:
    """Registry for dataset loaders."""
    
    _loaders: Dict[str, Type[BaseDatasetLoader]] = {
        "multiwoz": MultiWOZLoader,
        "sgd": SGDLoader,
        "custom": CustomLoader,
    }
    
    @classmethod
    def register(cls, name: str, loader_class: Type[BaseDatasetLoader]) -> None:
        """Register a new dataset loader."""
        cls._loaders[name] = loader_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseDatasetLoader]]:
        """Get a loader class by name."""
        return cls._loaders.get(name.lower())
    
    @classmethod
    def list_available(cls) -> Dict[str, str]:
        """List all available dataset loaders with descriptions."""
        return {
            name: loader_class.description
            for name, loader_class in cls._loaders.items()
        }
    
    @classmethod
    def create_loader(
        cls, 
        name: str, 
        data_dir: Optional[str] = None,
        **kwargs
    ) -> BaseDatasetLoader:
        """
        Create a loader instance by name.
        
        Args:
            name: Dataset name (multiwoz, sgd, custom)
            data_dir: Optional data directory
            **kwargs: Additional arguments for the loader
            
        Returns:
            Configured loader instance
        """
        loader_class = cls.get(name)
        if not loader_class:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {list(cls._loaders.keys())}"
            )
        return loader_class(data_dir=data_dir, **kwargs)


def get_loader(
    name: str,
    data_dir: Optional[str] = None,
    **kwargs
) -> BaseDatasetLoader:
    """Convenience function to get a dataset loader."""
    return DatasetRegistry.create_loader(name, data_dir, **kwargs)
