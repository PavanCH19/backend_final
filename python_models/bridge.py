#!/usr/bin/env python3
"""
Bridge script for executing Python functions from Node.js
Receives JSON via stdin, executes requested function, returns JSON via stdout
"""

import json
import sys
import os
import traceback
import importlib.util
from pathlib import Path


def load_module_from_path(script_path):
    """Dynamically load a Python module from a file path"""
    script_path = Path(script_path).resolve()
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # Add the script's directory to sys.path so it can import local modules
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Load the module dynamically
    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def execute_function(script_path, function_name, data):
    """Execute a function from a Python script with given data"""
    try:
        # Load the module from the script path
        module = load_module_from_path(script_path)
        
        # Get the function
        if not hasattr(module, function_name):
            available = [name for name in dir(module) if not name.startswith('_')]
            raise AttributeError(
                f"Function '{function_name}' not found in {script_path}. "
                f"Available functions: {', '.join(available)}"
            )
        
        func = getattr(module, function_name)
        
        if not callable(func):
            raise TypeError(f"'{function_name}' is not callable")
        
        # Execute the function
        result = func(data)
        
        return {
            "success": True,
            "result": result
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }


def main():
    """Main entry point - read request from stdin, execute, write response to stdout"""
    try:
        # Read the request from stdin
        request_str = sys.stdin.read()
        
        if not request_str.strip():
            raise ValueError("Empty request received from stdin")
        
        request = json.loads(request_str)
        
        # Extract parameters
        script_path = request.get("script_path")
        function_name = request.get("function")
        data = request.get("data", {})
        
        if not script_path:
            raise ValueError("Missing required field: script_path")
        
        if not function_name:
            raise ValueError("Missing required field: function")
        
        # Log to stderr (won't interfere with stdout JSON)
        print(f"[Bridge] Executing: {function_name} from {script_path}", 
              file=sys.stderr, flush=True)
        print(f"[Bridge] Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}", 
              file=sys.stderr, flush=True)
        
        # Execute the function
        response = execute_function(script_path, function_name, data)
        
        # Write response to stdout as JSON
        print(json.dumps(response, default=str), flush=True)
        
        # Exit with appropriate code
        sys.exit(0 if response["success"] else 1)
    
    except json.JSONDecodeError as e:
        error_response = {
            "success": False,
            "error": f"Invalid JSON in request: {str(e)}",
            "error_type": "JSONDecodeError"
        }
        print(json.dumps(error_response), flush=True)
        sys.exit(1)
    
    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_response), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()