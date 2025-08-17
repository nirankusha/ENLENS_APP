# -*- coding: utf-8 -*-
# colab_launcher.py - Simplified Colab-friendly launcher
import subprocess
import sys
import time
import threading
from google.colab import output

def launch_streamlit_colab(app="app_flexiconc.py", port=8501):
    """
    Launch Streamlit with Colab's built-in port forwarding
    More stable than ngrok for Colab usage
    """
    print("🚀 Starting Streamlit with Colab port forwarding...")
    
    # Start Streamlit in background
    cmd = [
        sys.executable, "-m", "streamlit", "run", app,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
        "--server.maxUploadSize", "50"  # Limit upload size
    ]
    
    # Start process
    proc = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )
    
    # Wait a moment for startup
    time.sleep(3)
    
    # Use Colab's port forwarding
    try:
        print(f"📱 Exposing on Colab iframe at port {port}")
        output.serve_kernel_port_as_iframe(port, height=800)
        print("✅ App should appear above!")
        
        # Simple log monitoring (non-blocking)
        def monitor_logs():
            try:
                for line in proc.stdout:
                    if "error" in line.lower() or "exception" in line.lower():
                        print(f"⚠️ {line.strip()}")
                    elif "running on" in line.lower():
                        print(f"✅ {line.strip()}")
            except:
                pass
        
        log_thread = threading.Thread(target=monitor_logs, daemon=True)
        log_thread.start()
        
    except Exception as e:
        print(f"❌ Error with Colab iframe: {e}")
        print(f"💡 Try accessing manually at: http://localhost:{port}")
    
    return proc

if __name__ == "__main__":
    # Simple usage
    proc = launch_streamlit_colab()
    
    # Keep alive
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("🛑 Stopping...")
        proc.terminate()
"""
Created on Sat Aug 16 20:51:14 2025

@author: niran
"""

