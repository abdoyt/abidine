"""
Model utilities for loading and managing autoencoder models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class ModelLoader:
    """Utility class for loading denoising autoencoder models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self, checkpoint_path: str, 
                       model_class: Optional[nn.Module] = None,
                       model_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load model checkpoint with comprehensive error handling.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_class: Model class to instantiate
            model_config: Model configuration parameters
            
        Returns:
            Dictionary containing model, optimizer, and training info
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Full training checkpoint
                    return {
                        'model_state_dict': checkpoint['model_state_dict'],
                        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
                        'epoch': checkpoint.get('epoch'),
                        'loss': checkpoint.get('loss'),
                        'config': checkpoint.get('config'),
                        'model_class': model_class,
                        'metadata': checkpoint.get('metadata', {})
                    }
                elif 'state_dict' in checkpoint:
                    # Alternative checkpoint format
                    return {
                        'model_state_dict': checkpoint['state_dict'],
                        'optimizer_state_dict': checkpoint.get('optimizer'),
                        'epoch': checkpoint.get('epoch'),
                        'loss': checkpoint.get('loss'),
                        'config': checkpoint.get('config'),
                        'model_class': model_class,
                        'metadata': checkpoint.get('metadata', {})
                    }
                else:
                    # Assume it's a state dict
                    return {
                        'model_state_dict': checkpoint,
                        'optimizer_state_dict': None,
                        'epoch': None,
                        'loss': None,
                        'config': model_config,
                        'model_class': model_class,
                        'metadata': {}
                    }
            else:
                # Assume it's a direct state dict
                return {
                    'model_state_dict': checkpoint,
                    'optimizer_state_dict': None,
                    'epoch': None,
                    'loss': None,
                    'config': model_config,
                    'model_class': model_class,
                    'metadata': {}
                }
        
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
            raise
    
    def create_model(self, model_config: Optional[Dict] = None) -> nn.Module:
        """
        Create a model instance based on configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Model instance
        """
        if model_config is None:
            model_config = {
                'input_channels': 3,
                'model_type': 'simple_autoencoder'
            }
        
        model_type = model_config.get('model_type', 'simple_autoencoder')
        
        if model_type == 'simple_autoencoder':
            return self._create_simple_autoencoder(model_config)
        elif model_type == 'resnet_autoencoder':
            return self._create_resnet_autoencoder(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_simple_autoencoder(self, config: Dict) -> nn.Module:
        """Create a simple autoencoder model."""
        input_channels = config.get('input_channels', 3)
        
        # Simple encoder-decoder architecture
        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_channels: int = 3):
                super(SimpleAutoencoder, self).__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True)
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, input_channels, 3, stride=2, padding=1, output_padding=1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return SimpleAutoencoder(input_channels)
    
    def _create_resnet_autoencoder(self, config: Dict) -> nn.Module:
        """Create a ResNet-based autoencoder model."""
        # This would implement a more sophisticated model
        # For now, return the simple one as placeholder
        self.logger.warning("ResNet autoencoder not implemented, using simple autoencoder")
        return self._create_simple_autoencoder(config)
    
    def load_model_with_checkpoint(self, checkpoint_path: str,
                                 model_config: Optional[Dict] = None) -> nn.Module:
        """
        Load a complete model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_config: Model configuration
            
        Returns:
            Loaded model ready for inference
        """
        # Load checkpoint data
        checkpoint_data = self.load_checkpoint(checkpoint_path, model_config=model_config)
        
        # Create model
        model = self.create_model(checkpoint_data.get('config', model_config))
        
        # Load state dict
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        return model


def save_model_config(config: Dict, output_path: str) -> None:
    """
    Save model configuration to JSON file.
    
    Args:
        config: Model configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(config_path: str) -> Dict:
    """
    Load model configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Model configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_model_config(config: Dict) -> bool:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['input_channels', 'model_type']
    
    for key in required_keys:
        if key not in config:
            return False
    
    valid_model_types = ['simple_autoencoder', 'resnet_autoencoder']
    if config['model_type'] not in valid_model_types:
        return False
    
    return True