#!/usr/bin/env python3
"""
Simple script to upload package to PyPI using urllib.
"""

import urllib.request
import urllib.parse
import base64
import os
import json

def upload_to_pypi():
    # Read the token
    with open('.pypi_token', 'r') as f:
        token = f.read().strip()
    
    # Create basic auth header
    auth_string = f'__token__:{token}'
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
    
    # Upload URL
    upload_url = 'https://upload.pypi.org/legacy/'
    
    # Files to upload
    files = [
        'lrdbenchmark-2.1.8.tar.gz',
        'lrdbenchmark-2.1.8-py3-none-any.whl'
    ]
    
    for filename in files:
        if os.path.exists(filename):
            print(f"Uploading {filename}...")
            
            # Read file
            with open(filename, 'rb') as f:
                file_data = f.read()
            
            # Create request
            req = urllib.request.Request(
                upload_url,
                data=file_data,
                headers={
                    'Authorization': f'Basic {auth_b64}',
                    'Content-Type': 'application/octet-stream'
                }
            )
            
            try:
                response = urllib.request.urlopen(req)
                print(f"Successfully uploaded {filename}")
                print(f"Response: {response.read().decode()}")
            except urllib.error.HTTPError as e:
                print(f"Error uploading {filename}: {e}")
                print(f"Response: {e.read().decode()}")
        else:
            print(f"File {filename} not found")

if __name__ == "__main__":
    upload_to_pypi()
