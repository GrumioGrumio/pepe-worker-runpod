import runpod
import os
import subprocess

def debug_volume_mount():
    """Debug volume mounting issues"""
    try:
        debug_info = {}
        
        # Check if /workspace exists and is mounted
        debug_info["workspace_exists"] = os.path.exists("/workspace")
        debug_info["workspace_is_mount"] = os.path.ismount("/workspace")
        
        # List contents of /workspace if it exists
        if os.path.exists("/workspace"):
            try:
                contents = os.listdir("/workspace")
                debug_info["workspace_contents"] = contents
            except Exception as e:
                debug_info["workspace_contents"] = f"Error listing: {str(e)}"
        else:
            debug_info["workspace_contents"] = "Directory doesn't exist"
        
        # Check mount points
        try:
            result = subprocess.run(["mount"], capture_output=True, text=True, timeout=10)
            mount_output = result.stdout
            
            # Look for network volume mounts
            volume_mounts = []
            for line in mount_output.split('\n'):
                if 'workspace' in line.lower() or 'volume' in line.lower():
                    volume_mounts.append(line.strip())
            
            debug_info["volume_mounts"] = volume_mounts
            debug_info["all_mounts"] = mount_output.split('\n')[:10]  # First 10 mounts
            
        except Exception as e:
            debug_info["mount_check"] = f"Error: {str(e)}"
        
        # Check disk usage
        try:
            result = subprocess.run(["df", "-h"], capture_output=True, text=True, timeout=10)
            df_output = result.stdout
            debug_info["disk_usage"] = df_output.split('\n')[:10]  # First 10 lines
        except Exception as e:
            debug_info["disk_usage"] = f"Error: {str(e)}"
        
        # Check environment variables
        debug_info["env_vars"] = {
            "RUNPOD_VOLUME_MOUNT_PATH": os.environ.get("RUNPOD_VOLUME_MOUNT_PATH"),
            "RUNPOD_VOLUME_ID": os.environ.get("RUNPOD_VOLUME_ID"),
            "PWD": os.environ.get("PWD"),
            "HOME": os.environ.get("HOME")
        }
        
        # Check root directory structure
        try:
            root_contents = os.listdir("/")
            debug_info["root_contents"] = sorted(root_contents)
        except Exception as e:
            debug_info["root_contents"] = f"Error: {str(e)}"
        
        # Check if we can create directories
        test_paths = ["/workspace", "/app", "/tmp/test"]
        debug_info["write_tests"] = {}
        
        for path in test_paths:
            try:
                os.makedirs(path, exist_ok=True)
                with open(f"{path}/test.txt", "w") as f:
                    f.write("test")
                debug_info["write_tests"][path] = "✅ Writable"
                os.remove(f"{path}/test.txt")
            except Exception as e:
                debug_info["write_tests"][path] = f"❌ Error: {str(e)}"
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

def handler(event):
    """Debug volume mounting issues"""
    print("🔍 VOLUME DEBUG HANDLER v17.0")
    print(f"👷 Worker: {event.get('workerId', 'unknown')}")
    
    try:
        # Get volume debug info
        debug_info = debug_volume_mount()
        
        # Analyze the situation
        analysis = {}
        
        if debug_info.get("workspace_exists"):
            if debug_info.get("workspace_is_mount"):
                analysis["volume_status"] = "✅ Volume properly mounted"
            else:
                analysis["volume_status"] = "⚠️ /workspace exists but not mounted as volume"
        else:
            analysis["volume_status"] = "❌ /workspace doesn't exist - volume not mounted"
        
        # Check if volume has contents
        contents = debug_info.get("workspace_contents", [])
        if isinstance(contents, list) and "comfyui" in contents:
            analysis["comfyui_status"] = "✅ ComfyUI found in volume"
        else:
            analysis["comfyui_status"] = "❌ ComfyUI not found in volume"
        
        # Check worker location
        volume_mounts = debug_info.get("volume_mounts", [])
        if volume_mounts:
            analysis["mount_status"] = f"✅ Found {len(volume_mounts)} volume mounts"
        else:
            analysis["mount_status"] = "❌ No volume mounts detected"
        
        return {
            "message": "🔍 Volume Debug Analysis",
            "worker_id": event.get("workerId", "unknown"),
            "debug_info": debug_info,
            "analysis": analysis,
            "recommendations": [
                "Check if volume is attached to endpoint in RunPod dashboard",
                "Verify volume and endpoint are in same datacenter (US-TX-3)",
                "Check if mount path is correct (/workspace)",
                "Try recreating the endpoint with volume attached",
                "Ensure worker gets assigned to correct datacenter"
            ],
            "next_steps": "If volume not mounted, check endpoint configuration in RunPod dashboard"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Debug handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting Volume Debug Handler...")
    runpod.serverless.start({'handler': handler})
