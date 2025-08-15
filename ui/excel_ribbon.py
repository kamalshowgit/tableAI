"""
Excel-like ribbon interface for AI Excel Assistant.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFrame, QGroupBox, QGridLayout,
                             QToolButton, QMenu, QAction, QFileDialog, QMessageBox,
                             QMenuBar, QMainWindow, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap

class ExcelRibbon(QWidget):
    """Excel-like ribbon interface with Windows-style dropdown menus."""
    
    # Signals
    fileOpenRequested = pyqtSignal()
    fileSaveRequested = pyqtSignal()
    fileSaveAsRequested = pyqtSignal()
    analyzeDataRequested = pyqtSignal()
    toolsRequested = pyqtSignal()
    testComplexQueryRequested = pyqtSignal()
    helpRequested = pyqtSignal()
    aiSearchRequested = pyqtSignal(str)
    exportCsvRequested = pyqtSignal()
    exportExcelRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the ribbon interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create menu bar style ribbon
        self.create_menu_ribbon()
        
        layout.addWidget(self.menu_ribbon)
        
    def create_menu_ribbon(self):
        """Create Windows-style dropdown menu ribbon."""
        self.menu_ribbon = QWidget()
        menu_layout = QHBoxLayout(self.menu_ribbon)
        menu_layout.setContentsMargins(5, 5, 5, 5)
        menu_layout.setSpacing(10)
        
        # Home dropdown menu
        home_menu = self.create_dropdown_menu("Home", [
            ("Paste", "Paste", None),
            ("Copy", "Copy", None),
            ("Cut", "Cut", None),
            ("", "", None),  # Separator
            ("Sort", "Sort", None),
            ("Filter", "Filter", None),
            ("Remove Duplicates", "Remove Duplicates", None),
            ("", "", None),  # Separator
            ("Analyze Data", "Analyze Data", self.analyzeDataRequested.emit)
        ])
        menu_layout.addWidget(home_menu)
        
        # File dropdown menu
        file_menu = self.create_dropdown_menu("File", [
            ("Open", "Open", self.fileOpenRequested.emit),
            ("Save", "Save", self.fileSaveRequested.emit),
            ("Save As", "Save As", self.fileSaveAsRequested.emit),
            ("", "", None),  # Separator
            ("Export CSV", "Export CSV", self.exportCsvRequested.emit),
            ("Export Excel", "Export Excel", self.exportExcelRequested.emit)
        ])
        menu_layout.addWidget(file_menu)
        
        # AI Search Box
        search_widget = self.create_search_box()
        menu_layout.addWidget(search_widget)
        
        # Tools dropdown menu
        tools_menu = self.create_dropdown_menu("Tools", [
            ("AI Analysis", "AI Analysis", self.toolsRequested.emit),
            ("Test Complex Query", "Test Complex Query", self.testComplexQueryRequested.emit),
            ("Validate Data", "Validate Data", None),
            ("Clean Data", "Clean Data", None)
        ])
        menu_layout.addWidget(tools_menu)
        
        # Help dropdown menu
        help_menu = self.create_dropdown_menu("Help", [
            ("Function Reference", "Function Reference", self.helpRequested.emit),
            ("User Guide", "User Guide", None),
            ("About", "About", None)
        ])
        menu_layout.addWidget(help_menu)
        
        # Add stretch to push menus to the left
        menu_layout.addStretch()
        
    def create_dropdown_menu(self, title: str, actions: list) -> QWidget:
        """Create a Windows-style dropdown menu button."""
        menu_widget = QWidget()
        menu_layout = QVBoxLayout(menu_widget)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(2)
        
        # Menu button
        menu_btn = QPushButton(title)
        menu_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
                min-width: 80px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #a0a0a0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        
        # Create dropdown menu
        dropdown_menu = QMenu(menu_btn)
        dropdown_menu.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 2px;
                color: #333333;
                background-color: transparent;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background-color: #e0e0e0;
                margin: 5px 0px;
            }
        """)
        
        # Add actions to menu
        for icon_text, action_text, callback in actions:
            if action_text == "" and icon_text == "":
                # Separator
                dropdown_menu.addSeparator()
            else:
                action = QAction(action_text, self)
                if callback:
                    action.triggered.connect(callback)
                dropdown_menu.addAction(action)
        
        # Connect button to menu
        menu_btn.setMenu(dropdown_menu)
        
        # Add button to layout
        menu_layout.addWidget(menu_btn)
        
        return menu_widget
    
    def create_search_box(self) -> QWidget:
        """Create an AI search box for quick questions."""
        search_widget = QWidget()
        search_layout = QVBoxLayout(search_widget)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(2)
        
        # Search label
        search_label = QLabel("AI Search")
        search_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                font-weight: bold;
                color: #495057;
                text-align: center;
                margin: 2px;
            }
        """)
        search_layout.addWidget(search_label)
        
        # Search input box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Ask AI a question...")
        self.search_input.setMaximumWidth(200)
        self.search_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 11px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #217346;
                border-width: 2px;
            }
        """)
        
        # Connect enter key to search
        self.search_input.returnPressed.connect(self.perform_search)
        search_layout.addWidget(self.search_input)
        
        return search_widget
    
    def perform_search(self):
        """Perform AI search when Enter is pressed."""
        query = self.search_input.text().strip()
        if query:
            self.aiSearchRequested.emit(query)
            self.search_input.clear()
