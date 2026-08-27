import os
import sys
import time
import atexit
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Omniscient...")

    python_exe = sys.executable
    npm_exe = shutil.which("npm")
    
    if not npm_exe:
        logger.error("Could not find 'npm' in PATH. Please ensure Node.js is installed.")
        sys.exit(1)

    # Launch tracking engine
    engine_process = subprocess.Popen([python_exe, "-m", "engine.main"])
    logger.info("Tracking engine started.")

    # Launch UI overlay
    overlay_process = subprocess.Popen([npm_exe, "start"], cwd="overlay")
    logger.info("UI overlay started.")

    def cleanup():
        logger.info("\nShutting down Omniscient...")
        if engine_process.poll() is None:
            engine_process.terminate()
            try:
                engine_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                engine_process.kill()
                
        if overlay_process.poll() is None:
            overlay_process.terminate()
            try:
                overlay_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                overlay_process.kill()

    atexit.register(cleanup)

    try:
        # Keep running while both processes are active
        while True:
            engine_code = engine_process.poll()
            if engine_code is not None:
                logger.error(f"Tracking engine exited unexpectedly (code {engine_code}).")
                break
                
            overlay_code = overlay_process.poll()
            if overlay_code is not None:
                logger.info(f"UI overlay closed (code {overlay_code}).")
                break
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal.")
        
    sys.exit(0)

if __name__ == "__main__":
    main()
