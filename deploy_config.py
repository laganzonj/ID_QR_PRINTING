#!/usr/bin/env python3
"""
Deployment configuration and health check for QR ID printing application.
This script helps ensure the application can start properly in deployment.
"""

import os
import sys
import importlib
import traceback

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'flask',
        'pandas', 
        'qrcode',
        'PIL',
        'reportlab',
        'cryptography',
        'psutil'
    ]
    
    optional_packages = {
        'flask_mail': 'Email functionality',
        'apscheduler': 'Background task scheduling'
    }
    
    missing_required = []
    missing_optional = []
    
    print("Checking required dependencies...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"✗ {package} - MISSING")
    
    print("\nChecking optional dependencies...")
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            missing_optional.append(package)
            print(f"○ {package} - {description} (optional, will be disabled)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Optional packages not available: {', '.join(missing_optional)}")
    
    print("\n✅ All required dependencies are available!")
    return True

def check_memory():
    """Check available memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        
        print(f"\nMemory Status:")
        print(f"Total: {mem.total / (1024**3):.1f} GB")
        print(f"Available: {available_mb:.1f} MB")
        print(f"Used: {mem.percent:.1f}%")
        
        if available_mb < 100:
            print("⚠️  Low memory warning: Less than 100MB available")
            return False
        
        print("✅ Memory status OK")
        return True
        
    except Exception as e:
        print(f"❌ Memory check failed: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        'data',
        'data/participant_list',
        'data/id_templates', 
        'data/qr_codes',
        'static',
        'templates'
    ]
    
    print("\nChecking directories...")
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            missing_dirs.append(dir_path)
            print(f"✗ {dir_path} - MISSING")
            
            # Try to create missing directories
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  → Created {dir_path}")
            except Exception as e:
                print(f"  → Failed to create {dir_path}: {e}")
    
    if missing_dirs:
        print(f"\n⚠️  Some directories were missing but have been created")
    
    print("✅ Directory structure OK")
    return True

def test_app_import():
    """Test if the Flask app can be imported"""
    print("\nTesting app import...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the app
        from app import app
        print("✅ App imported successfully")
        
        # Test basic app configuration
        if app.config:
            print("✅ App configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def create_minimal_config():
    """Create minimal configuration files if they don't exist"""
    print("\nCreating minimal configuration...")
    
    # Create active_config.json if it doesn't exist
    config_path = "data/active_config.json"
    if not os.path.exists(config_path):
        try:
            import json
            minimal_config = {
                "active_dataset": "",
                "active_template": "default_template.png"
            }
            with open(config_path, 'w') as f:
                json.dump(minimal_config, f, indent=2)
            print(f"✓ Created {config_path}")
        except Exception as e:
            print(f"✗ Failed to create {config_path}: {e}")
    
    # Create .ready file
    ready_path = "data/.ready"
    if not os.path.exists(ready_path):
        try:
            with open(ready_path, 'w') as f:
                f.write("ready")
            print(f"✓ Created {ready_path}")
        except Exception as e:
            print(f"✗ Failed to create {ready_path}: {e}")
    
    print("✅ Configuration files ready")

def main():
    """Run all deployment checks"""
    print("🚀 QR ID Printing App - Deployment Health Check")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Memory", check_memory), 
        ("Directories", check_directories),
        ("Configuration", create_minimal_config),
        ("App Import", test_app_import)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name} Check:")
        print("-" * 30)
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! App should deploy successfully.")
        print("\n💡 To enable email features, install Flask-Mail:")
        print("   pip install Flask-Mail==0.9.1")
        print("\n💡 To enable background tasks, install APScheduler:")
        print("   pip install APScheduler==3.10.4")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
