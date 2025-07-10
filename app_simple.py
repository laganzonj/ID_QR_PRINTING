#!/usr/bin/env python3
"""
Simplified deployment-safe version of the QR ID printing application.
This version removes threading and complex background processing for deployment stability.
"""

import os
import sys

# Set up environment for deployment
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('ENABLE_SCHEDULER', 'false')
os.environ.setdefault('ENABLE_BACKGROUND_QR', 'false')

# Import the main app
try:
    from app import app
    
    # Disable threading features for deployment
    app.config['THREADING'] = False
    
    print("QR ID Printing App - Deployment Mode")
    print("Features:")
    print("  - Core QR scanning and printing: ✓")
    print("  - On-demand QR generation: ✓")
    print("  - Background threading: ✗ (disabled for deployment)")
    print("  - Email features: ✗ (optional)")
    print("  - Scheduler: ✗ (disabled)")
    
    if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5000))
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=False,  # Disable threading
            processes=1
        )
        
except Exception as e:
    print(f"Failed to start app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
