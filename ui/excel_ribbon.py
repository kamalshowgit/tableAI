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
        
        # Set ribbon background and styling
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
            }
        """)
        
    def create_menu_ribbon(self):
        """Create Windows-style dropdown menu ribbon."""
        self.menu_ribbon = QWidget()
        self.menu_ribbon.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #217346, stop:1 #1a5c3a);
                border-bottom: 2px solid #155a2e;
            }
        """)
        menu_layout = QHBoxLayout(self.menu_ribbon)
        menu_layout.setContentsMargins(15, 12, 15, 12)
        menu_layout.setSpacing(15)
        
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
                background-color: rgba(255, 255, 255, 0.9);
                color: #217346;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                padding: 10px 18px;
                font-weight: bold;
                font-size: 12px;
                min-width: 90px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 1.0);
                border-color: rgba(255, 255, 255, 0.6);
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.8);
                transform: translateY(0px);
            }
        """)
        
        # Create dropdown menu
        dropdown_menu = QMenu(menu_btn)
        dropdown_menu.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                border: 2px solid #217346;
                border-radius: 8px;
                padding: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
            QMenu::item {
                padding: 10px 24px;
                border-radius: 6px;
                color: #495057;
                background-color: transparent;
                font-size: 12px;
                font-weight: 500;
            }
            QMenu::item:selected {
                background-color: #217346;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background-color: #e9ecef;
                margin: 8px 0px;
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
        search_layout.setSpacing(3)
        
        # Search label
        search_label = QLabel("AI Search")
        search_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                font-weight: bold;
                color: #217346;
                text-align: center;
                margin: 2px;
                background-color: #f8f9fa;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #e9ecef;
            }
        """)
        search_layout.addWidget(search_label)
        
        # Search input box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Ask AI a question...")
        self.search_input.setMaximumWidth(220)
        self.search_input.setMinimumHeight(28)
        self.search_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                background-color: white;
                color: #333333;
                font-weight: 500;
            }
            QLineEdit:focus {
                border-color: #217346;
                border-width: 2px;
                background-color: #f8f9fa;
            }
            QLineEdit::placeholder {
                color: #6c757d;
                font-style: italic;
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
