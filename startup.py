#!/usr/bin/env python3
"""
Startup script for QR ID printing application.
Handles graceful startup with error recovery and minimal dependencies.
"""

import os
import sys
import gc
import time

def setup_environment():
    """Setup environment variables for deployment"""
    # Memory optimization
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    os.environ.setdefault('MALLOC_ARENA_MAX', '2')
    
    # App configuration
    os.environ.setdefault('MEMORY_LIMIT_MB', '150')
    os.environ.setdefault('ENABLE_SCHEDULER', 'false')  # Disable by default
    
    print("Environment configured for deployment")

def create_directories():
    """Create required directories"""
    dirs = [
        'data',
        'data/participant_list',
        'data/id_templates',
        'data/qr_codes',
        'static/id_templates'
    ]
    
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create {dir_path}: {e}")

def create_config_files():
    """Create minimal configuration files"""
    try:
        import json
        
        # Create active_config.json
        config_path = "data/active_config.json"
        if not os.path.exists(config_path):
            config = {
                "active_dataset": "",
                "active_template": "default_template.png"
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create .ready file
        ready_path = "data/.ready"
        if not os.path.exists(ready_path):
            with open(ready_path, 'w') as f:
                f.write("ready")
        
        print("Configuration files created")
        
    except Exception as e:
        print(f"Warning: Could not create config files: {e}")

def optimize_memory():
    """Optimize memory usage for deployment"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Set garbage collection thresholds for memory efficiency
        gc.set_threshold(700, 10, 10)
        
        print("Memory optimization applied")
        
    except Exception as e:
        print(f"Warning: Memory optimization failed: {e}")

def start_app():
    """Start the Flask application with error handling"""
    try:
        print("Starting QR ID Printing Application...")
        
        # Setup environment
        setup_environment()
        
        # Create directories and config
        create_directories()
        create_config_files()
        
        # Optimize memory
        optimize_memory()
        
        # Import and start the app
        from app import app
        
        # Get port from environment
        port = int(os.environ.get('PORT', 5000))
        
        print(f"Application starting on port {port}")
        print("Features available:")
        
        # Check available features
        try:
            from app import MAIL_AVAILABLE
            print(f"  - Email: {'✓' if MAIL_AVAILABLE else '✗'}")
        except:
            print("  - Email: ✗")
        
        try:
            from app import SCHEDULER_AVAILABLE
            print(f"  - Background Tasks: {'✓' if SCHEDULER_AVAILABLE else '✗'}")
        except:
            print("  - Background Tasks: ✗")
        
        # Start the app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True
        )
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Some dependencies may be missing. The app will run with limited features.")
        
        # Try to start with minimal features
        try:
            from app import app
            port = int(os.environ.get('PORT', 5000))
            app.run(host='0.0.0.0', port=port, debug=False)
        except Exception as e2:
            print(f"Failed to start app: {e2}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Startup Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_app()
