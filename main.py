#!/usr/bin/env python3
"""
Main entry point for the AI Excel Assistant PyQt application.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.main_window import ExcelAIAssistant

def main():
    """Main application function."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application properties
    app.setApplicationName("AI Excel Assistant")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("Local AI")
    
    # Create and show the main window
    window = ExcelAIAssistant()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
