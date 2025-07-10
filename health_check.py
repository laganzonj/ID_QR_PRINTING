#!/usr/bin/env python3
"""
Health check script for deployment troubleshooting.
Run this to diagnose deployment issues.
"""

import os
import sys
import traceback

def check_python_version():
    """Check Python version compatibility"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def check_basic_imports():
    """Check if basic imports work"""
    try:
        import flask
        print(f"✅ Flask {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import qrcode
        print(f"✅ QRCode available")
    except ImportError as e:
        print(f"❌ QRCode import failed: {e}")
        return False
    
    return True

def check_optional_imports():
    """Check optional imports"""
    optional_packages = {
        'flask_mail': 'Email functionality',
        'apscheduler': 'Background scheduling'
    }
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"○ {package} - {description} (optional, disabled)")

def check_directories():
    """Check required directories"""
    dirs = ['data', 'static', 'templates']
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"  → Created {dir_name}/")
            except Exception as e:
                print(f"  → Failed to create {dir_name}/: {e}")

def check_app_import():
    """Check if app can be imported"""
    try:
        print("Testing app import...")
        
        # Set deployment environment
        os.environ['ENABLE_SCHEDULER'] = 'false'
        os.environ['ENABLE_BACKGROUND_QR'] = 'false'
        
        from app import app
        print("✅ App imported successfully")
        
        # Test basic app properties
        if hasattr(app, 'config'):
            print("✅ App configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def check_memory():
    """Check available memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Memory: {mem.available / (1024**2):.1f}MB available ({100-mem.percent:.1f}% free)")
        
        if mem.available < 100 * 1024 * 1024:  # 100MB
            print("⚠️  Low memory warning")
        else:
            print("✅ Memory OK")
            
    except ImportError:
        print("○ psutil not available, cannot check memory")
    except Exception as e:
        print(f"❌ Memory check failed: {e}")

def main():
    """Run all health checks"""
    print("🏥 QR ID Printing App - Health Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Basic Imports", check_basic_imports),
        ("Optional Imports", check_optional_imports),
        ("Directories", check_directories),
        ("Memory", check_memory),
        ("App Import", check_app_import)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        print("-" * 30)
        try:
            result = check_func()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed!")
        print("\n🚀 Ready for deployment")
    else:
        print("❌ Some checks failed")
        print("\n💡 Try running: pip install -r requirements.txt")
    
    print("\n📝 Deployment Notes:")
    print("  - Threading disabled for deployment stability")
    print("  - Background QR generation uses synchronous fallback")
    print("  - Email features optional (Flask-Mail)")
    print("  - Scheduler disabled by default")

if __name__ == "__main__":
    main()
