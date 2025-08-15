"""
Dialog classes for AI Excel Assistant PyQt application.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QLabel, QPushButton, QCheckBox, QScrollArea,
                             QFrame, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import pandas as pd

class StepApprovalDialog(QDialog):
    """Dialog for step approval and preview."""
    
    def __init__(self, step_name: str, description: str, code: str, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.description = description
        self.code = code
        self.approved = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle(f"Step Approval: {self.step_name}")
        self.setGeometry(300, 200, 700, 500)
        self.setModal(True)
        
        # Set window properties
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        
        layout = QVBoxLayout(self)
        
        # Step name
        name_label = QLabel(f"Step: {self.step_name}")
        name_label.setStyleSheet("""
            QLabel {
                font-size: 18px; 
                font-weight: bold; 
                color: #217346; 
                margin: 10px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 8px;
            }
        """)
        layout.addWidget(name_label)
        
        # Description
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold; margin-top: 15px; font-size: 14px;")
        layout.addWidget(desc_label)
        
        desc_text = QLabel(self.description)
        desc_text.setWordWrap(True)
        desc_text.setStyleSheet("""
            QLabel {
                margin: 5px; 
                padding: 15px; 
                background-color: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 8px;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(desc_text)
        
        # Code preview
        code_label = QLabel("Generated Code:")
        code_label.setStyleSheet("font-weight: bold; margin-top: 20px; font-size: 14px;")
        layout.addWidget(code_label)
        
        code_text = QTextEdit()
        code_text.setPlainText(self.code)
        code_text.setMaximumHeight(150)
        code_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d3748;
                color: #00ff88;
                border: 1px solid #4a5568;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 15px;
                selection-background-color: #4a5568;
            }
        """)
        code_text.setReadOnly(True)
        layout.addWidget(code_text)
        
        # Approval checkbox
        self.approve_checkbox = QCheckBox("Approve this step for execution")
        self.approve_checkbox.setChecked(True)
        self.approve_checkbox.setStyleSheet("""
            QCheckBox {
                margin: 20px 0; 
                font-weight: bold; 
                font-size: 14px;
                color: #217346;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        layout.addWidget(self.approve_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        approve_btn = QPushButton("Approve & Continue")
        approve_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #218838;
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        approve_btn.clicked.connect(self.accept)
        
        reject_btn = QPushButton("Reject Step")
        reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        reject_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(approve_btn)
        button_layout.addWidget(reject_btn)
        layout.addLayout(button_layout)
    
    def is_approved(self) -> bool:
        """Check if the step is approved."""
        return self.approve_checkbox.isChecked()

class ExecutionSummaryDialog(QDialog):
    """Dialog for showing execution summary."""
    
    def __init__(self, question: str, result_df: pd.DataFrame, operation_log: list, business_insight: str, parent=None):
        super().__init__(parent)
        self.question = question
        self.result_df = result_df
        self.operation_log = operation_log
        self.business_insight = business_insight
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Execution Summary")
        self.setGeometry(400, 300, 700, 600)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Summary header
        header_label = QLabel("Execution Summary")
        header_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #217346;
                margin: 15px;
                text-align: center;
            }
        """)
        layout.addWidget(header_label)
        
        # Summary text
        summary_text = self._generate_summary_text()
        summary_label = QLabel(summary_text)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("""
            QLabel {
                color: #333333;
                background-color: #ffffff;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.6;
                font-weight: normal;
            }
        """)
        
        # Scroll area for long summaries
        scroll = QScrollArea()
        scroll.setWidget(summary_label)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #f1f3f4;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #dadce0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #bdc1c6;
            }
        """)
        layout.addWidget(scroll)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #217346;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 100px;
                margin: 15px;
            }
            QPushButton:hover {
                background-color: #1a5c3a;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def _generate_summary_text(self) -> str:
        """Generate the summary text."""
        summary_text = f"AI Analysis Results\n\n"
        summary_text += f"Question: {self.question}\n\n"
        summary_text += f"Operations Performed:\n"
        for log in self.operation_log:
            summary_text += f"  • {log}\n"
        
        summary_text += f"\nResults Summary:\n"
        summary_text += f"  • Result rows: {len(self.result_df):,}\n"
        summary_text += f"  • Result columns: {len(self.result_df.columns)}\n"
        
        summary_text += f"\nBusiness Insight:\n"
        # Clean up markdown formatting from AI responses
        cleaned_insight = self._clean_markdown_formatting(self.business_insight)
        summary_text += cleaned_insight
        
        return summary_text
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Clean up markdown formatting from AI responses."""
        import re
        
        # Remove markdown bold formatting (**text** -> text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # Remove markdown italic formatting (*text* -> text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # Remove markdown code formatting (`text` -> text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Remove markdown headers (# Header -> Header)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove markdown links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text

class DataPreviewDialog(QDialog):
    """Dialog for showing data preview."""
    
    def __init__(self, data_preview: str, title: str = "Data Preview", parent=None):
        super().__init__(parent)
        self.data_preview = data_preview
        self.title = title
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle(self.title)
        self.setGeometry(300, 200, 800, 600)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #217346;
                margin: 10px;
                text-align: center;
            }
        """)
        layout.addWidget(title_label)
        
        # Data preview
        preview_text = QTextEdit()
        preview_text.setPlainText(self.data_preview)
        preview_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 15px;
            }
        """)
        preview_text.setReadOnly(True)
        layout.addWidget(preview_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class AIQuestionDialog(QDialog):
    """Dialog for AI question input."""
    
    questionSubmitted = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Ask AI Question")
        self.setGeometry(300, 200, 600, 400)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Ask a question about your data in natural language:")
        instructions.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(instructions)
        
        # Examples
        examples = QLabel("""
Examples:
• "Show me sales above $1000"
• "What is the average salary?"
• "Group by department and count employees"
• "Find all customers from New York"
        """)
        examples.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 10px;
                margin: 10px;
                font-size: 12px;
                color: #333333;
                font-weight: normal;
            }
        """)
        layout.addWidget(examples)
        
        # Question input
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Enter your question here...")
        self.question_input.setMaximumHeight(100)
        self.question_input.setStyleSheet("""
            QTextEdit {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTextEdit:focus {
                border-color: #217346;
            }
        """)
        layout.addWidget(self.question_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        submit_btn = QPushButton("Submit Question")
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #217346;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1a5c3a;
            }
        """)
        submit_btn.clicked.connect(self.submit_question)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(submit_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def submit_question(self):
        """Submit the question and close dialog."""
        question = self.question_input.toPlainText().strip()
        if question:
            self.questionSubmitted.emit(question)
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please enter a question.")
