#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def main():
    """Run administrative tasks."""
    # Kerrotaan Pythonille, että mikrobot_mcp löytyy webui-kansiosta
    current_path = os.path.dirname(os.path.abspath(__file__))
    webui_path = os.path.join(current_path, 'webui')
    if webui_path not in sys.path:
        sys.path.append(webui_path)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mikrobot_mcp.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed...?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()