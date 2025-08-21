import streamlit as st
from typing import Tuple, Dict, Any, List
import mimetypes
import requests
import os

def validate_pdf_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded PDF file with enhanced diagnostics.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if uploaded_file is None:
            return False, "No file uploaded"
        
        print(f"📄 Validating file: {uploaded_file.name}")
        print(f"📊 File size: {uploaded_file.size} bytes ({uploaded_file.size / (1024*1024):.2f} MB)")
        
        # Check file extension
        if not uploaded_file.name.lower().endswith('.pdf'):
            print(f"❌ Invalid extension: {uploaded_file.name}")
            return False, "File must be a PDF (.pdf extension)"
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        print(f"🔍 Detected MIME type: {mime_type}")
        if mime_type != 'application/pdf':
            print(f"❌ Invalid MIME type: {mime_type}")
            return False, "File must be a valid PDF document"
        
        # Check file size (more generous limit - 50MB)
        max_size = 50 * 1024 * 1024  # 50MB in bytes (matches Streamlit config)
        if uploaded_file.size > max_size:
            size_mb = uploaded_file.size / (1024 * 1024)
            print(f"❌ File too large: {size_mb:.1f} MB > 50 MB")
            return False, f"File size ({size_mb:.1f} MB) exceeds 50MB limit"
        
        # Check minimum size
        if uploaded_file.size < 100:  # Less than 100 bytes is likely not a valid PDF
            print(f"❌ File too small: {uploaded_file.size} bytes")
            return False, "File is too small to be a valid PDF"
        
        # Check PDF header
        try:
            file_content = uploaded_file.getvalue()
            if not file_content.startswith(b'%PDF'):
                print(f"❌ Invalid PDF header: {file_content[:10]}")
                return False, "File does not appear to be a valid PDF (invalid header)"
            else:
                print(f"✅ Valid PDF header detected")
        except Exception as header_error:
            print(f"❌ Error reading file header: {header_error}")
            return False, f"Error reading file: {str(header_error)}"
        
        print(f"✅ File validation successful")
        return True, ""
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False, f"Validation error: {str(e)}"

def format_chat_message(message: Dict[str, str]) -> str:
    """
    Format chat message for display.
    
    Args:
        message: Message dictionary with 'role' and 'content' keys
        
    Returns:
        Formatted message string
    """
    role = message.get('role', 'unknown')
    content = message.get('content', '')
    
    if role == 'user':
        return f"**You:** {content}"
    elif role == 'assistant':
        return f"**Assistant:** {content}"
    else:
        return f"**{role.title()}:** {content}"

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def clean_filename(filename: str) -> str:
    """
    Clean filename for safe display.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    import re
    
    # Remove or replace unsafe characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove excessive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores and spaces
    cleaned = cleaned.strip('_ ')
    
    return cleaned if cleaned else "unnamed_file"

def get_file_info(uploaded_file) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dictionary with file information
    """
    if uploaded_file is None:
        return {}
    
    return {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'size_formatted': format_file_size(uploaded_file.size),
        'type': uploaded_file.type,
        'cleaned_name': clean_filename(uploaded_file.name)
    }

def create_download_link(data: str, filename: str, text: str = "Download") -> str:
    """
    Create a download link for text data.
    
    Args:
        data: Text data to download
        filename: Suggested filename
        text: Link text
        
    Returns:
        HTML download link
    """
    import base64
    
    # Encode data
    b64_data = base64.b64encode(data.encode()).decode()
    
    # Create download link
    href = f'<a href="data:text/plain;base64,{b64_data}" download="{filename}">{text}</a>'
    
    return href

@st.cache_data
def get_system_info() -> Dict[str, str]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    return {
        'platform': platform.system(),
        'python_version': sys.version,
        'streamlit_version': st.__version__
    }

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def format_number(number: float, decimal_places: int = 2) -> str:
    """
    Format number with specified decimal places and thousand separators.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted number string
    """
    try:
        return f"{number:,.{decimal_places}f}"
    except (ValueError, TypeError):
        return str(number)

def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """
    Extract simple keywords from text (basic implementation).
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    import re
    from collections import Counter
    
    # Simple keyword extraction - remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Extract words (alphanumeric only)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count frequency and return most common
    word_counts = Counter(filtered_words)
    
    return [word for word, count in word_counts.most_common(max_keywords)]



