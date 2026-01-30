#!/usr/bin/env python3
"""
Configuration manager tool - automatically modify device configuration in config.yaml

This tool helps automatically adjust GPU device allocation without manual edits to the config file.
"""

import yaml
import argparse
from pathlib import Path


class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    def load_config(self):
        """Load configuration file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config):
        """Save configuration file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def update_devices(self, base_device: str = None, draft_device: str = None):
        """Update device configuration"""
        config = self.load_config()
        
        if base_device is not None:
            config['device_base'] = base_device
            print(f"✅ Set device_base = {base_device}")
        
        if draft_device is not None:
            config['device_draft'] = draft_device
            print(f"✅ Set device_draft = {draft_device}")
        
        self.save_config(config)
        print(f"💾 Configuration saved to {self.config_path}")
    
    def show_current_config(self):
        """Show current device configuration"""
        config = self.load_config()
        
        print("📋 Current device configuration:")
        print(f"   Base device: {config.get('device_base', 'Not set')}")
        print(f"   Draft device: {config.get('device_draft', 'Not set')}")
        print(f"   Communication port: {config.get('communication_port', 'Not set')}")


def main():
    parser = argparse.ArgumentParser(description="Configuration manager tool")
    
    parser.add_argument('--base-device', type=str,
                       help='Set GPU device for base model, e.g.: cuda:0')
    
    parser.add_argument('--draft-device', type=str,
                       help='Set GPU device for draft model, e.g.: cuda:1')
    
    parser.add_argument('--show', action='store_true',
                       help='Show current configuration')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path (default: config.yaml)')
    
    args = parser.parse_args()
    
    try:
        manager = ConfigManager(args.config)
        
        if args.show:
            manager.show_current_config()
        
        if args.base_device or args.draft_device:
            manager.update_devices(args.base_device, args.draft_device)
        
            if not args.show and not args.base_device and not args.draft_device:
                print("❓ Use --help to see usage")
                manager.show_current_config()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()