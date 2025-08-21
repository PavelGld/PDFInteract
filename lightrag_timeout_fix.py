"""
LightRAG timeout fix for v1.3.7+ httpx timeout bug
This module patches the OpenRouter client to use longer timeouts for complex queries
"""

import os
import sys
import httpx
from typing import Optional

def patch_lightrag_timeouts():
    """
    Patch LightRAG internal httpx clients to use longer timeouts
    This fixes the httpx.ReadTimeout issues in v1.3.7+
    """
    try:
        # Override httpx timeout defaults globally
        original_timeout = httpx.Timeout
        
        class ExtendedTimeout(original_timeout):
            def __init__(self, timeout=None, **kwargs):
                # Force long timeouts for LightRAG operations
                if timeout is None:
                    timeout = httpx.Timeout(
                        connect=60.0,   # 1 minute connect
                        read=600.0,     # 10 minutes read (for complex graph ops)
                        write=120.0,    # 2 minutes write  
                        pool=None       # No pool timeout
                    )
                super().__init__(timeout, **kwargs)
        
        # Monkey patch httpx.Timeout
        httpx.Timeout = ExtendedTimeout
        
        print("✓ Applied LightRAG timeout fix")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not apply timeout fix: {e}")
        return False

def setup_environment_timeouts():
    """
    Set environment variables that LightRAG might respect
    """
    os.environ["TIMEOUT"] = "600"  # 10 minutes
    os.environ["READ_TIMEOUT"] = "600"
    os.environ["CONNECT_TIMEOUT"] = "60"
    os.environ["HTTPX_TIMEOUT"] = "600"
    os.environ["LLM_REQUEST_TIMEOUT"] = "600"
    print("✓ Set extended timeout environment variables")

if __name__ == "__main__":
    patch_lightrag_timeouts()
    setup_environment_timeouts()