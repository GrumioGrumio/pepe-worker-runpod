import runpod
import json
import time

def handler(event):
    """Simple test handler first"""
    print("🐸 Pepe Worker Test Started!")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Received prompt: {prompt}")
        
        # Simulate processing time
        time.sleep(2)
        
        # Return success response
        return {
            "message": f"Pepe worker received: {prompt}",
            "status": "success",
            "test": "Basic handler working!",
            "next_step": "Will add ComfyUI + FLUX + LoRA once basic test works"
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == '__main__':
    print("🚀 Starting Pepe Test Worker...")
    runpod.serverless.start({'handler': handler})
