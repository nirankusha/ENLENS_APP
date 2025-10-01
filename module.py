# -*- coding: utf-8 -*-
import importlib.util
import sys
from google.colab import files
import tempfile
import os

def upload_and_load_module(module_name, prompt=None):
    """Upload and load module directly from file content"""
    if prompt:
        print(prompt)
    else:
        print(f"Upload {module_name}.py")

    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # Create a temporary file with the uploaded content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        # Decode bytes to string if necessary
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8')
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # Load module from temporary file
        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
        module = importlib.util.module_from_spec(spec)

        # Remove old module if it exists
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Execute the module
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        print(f"✅ Loaded {module_name} from upload")
        return module

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
"""
Created on Wed Aug 20 19:05:37 2025

@author: niran
"""

