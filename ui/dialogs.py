"""
Dialog classes for AI Excel Assistant PyQt application.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QLabel, QPushButton, QCheckBox, QScrollArea,
                             QFrame, QGridLayout, QMessageBox, QTableWidget,
                             QHeaderView, QWidget)
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

class MultiStepApprovalDialog(QDialog):
    """Dialog for multi-step operations with previews and approvals."""
    
    allStepsApproved = pyqtSignal(list)
    workflowCancelled = pyqtSignal()
    
    def __init__(self, df: pd.DataFrame, steps: list, parent=None):
        super().__init__(parent)
        self.df = df
        self.steps = steps
        self.current_step = 0
        self.approved_steps = []
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Multi-Step Operation Approval")
        self.setGeometry(400, 300, 900, 700)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Progress indicator
        progress_label = QLabel(f"Step {self.current_step + 1} of {len(self.steps)}")
        progress_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #217346;
                margin: 10px;
                text-align: center;
            }
        """)
        layout.addWidget(progress_label)
        
        # Current step info
        if self.steps:
            current_step_info = self.steps[self.current_step]
            step_name_label = QLabel(f"Operation: {current_step_info.get('name', 'Unknown')}")
            step_name_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: #495057;
                    margin: 10px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                }
            """)
            layout.addWidget(step_name_label)
            
            # Description
            desc_label = QLabel("Description:")
            desc_label.setStyleSheet("font-weight: bold; margin-top: 15px; font-size: 14px;")
            layout.addWidget(desc_label)
            
            desc_text = QLabel(current_step_info.get('description', 'No description available'))
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
            code_text.setPlainText(current_step_info.get('code', 'No code available'))
            code_text.setMaximumHeight(120)
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
            
            # Data preview
            preview_label = QLabel("Data Preview (First 10 rows):")
            preview_label.setStyleSheet("font-weight: bold; margin-top: 20px; font-size: 14px;")
            layout.addWidget(preview_label)
            
            # Create preview table
            self.preview_table = QTableWidget()
            self.preview_table.setMaximumHeight(200)
            self.preview_table.setStyleSheet("""
                QTableWidget {
                    gridline-color: #dee2e6;
                    background-color: white;
                    alternate-background-color: #f8f9fa;
                }
                QHeaderView::section {
                    background-color: #e9ecef;
                    padding: 6px;
                    border: 1px solid #dee2e6;
                    font-weight: bold;
                }
            """)
            self.preview_table.setAlternatingRowColors(True)
            self._update_preview_table()
            layout.addWidget(self.preview_table)
        
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
        
        if self.current_step < len(self.steps) - 1:
            next_btn = QPushButton("Approve & Continue to Next Step")
            next_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 14px;
                    min-width: 200px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """)
            next_btn.clicked.connect(self.next_step)
            button_layout.addWidget(next_btn)
        else:
            finish_btn = QPushButton("Approve & Finish All Steps")
            finish_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 14px;
                    min-width: 200px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """)
            finish_btn.clicked.connect(self.finish_all_steps)
            button_layout.addWidget(finish_btn)
        
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
        reject_btn.clicked.connect(self.reject_step)
        
        cancel_btn = QPushButton("Cancel All")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        cancel_btn.clicked.connect(self.cancel_workflow)
        
        button_layout.addWidget(reject_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _update_preview_table(self):
        """Update the preview table with current data."""
        if not self.steps or self.current_step >= len(self.steps):
            return
            
        # Get the preview data for current step
        preview_data = self.steps[self.current_step].get('preview_data', self.df)
        
        if preview_data is None or preview_data.empty:
            return
            
        # Update table dimensions
        rows = min(len(preview_data), 10)
        cols = min(len(preview_data.columns), 10)
        
        self.preview_table.setRowCount(rows)
        self.preview_table.setColumnCount(cols)
        
        # Set headers
        column_headers = list(preview_data.columns[:cols])
        self.preview_table.setHorizontalHeaderLabels(column_headers)
        
        # Populate table
        for i in range(rows):
            for j in range(cols):
                value = str(preview_data.iloc[i, j])
                if len(value) > 30:  # Truncate long values
                    value = value[:27] + "..."
                item = QTableWidgetItem(value)
                self.preview_table.setItem(i, j, item)
        
        # Resize columns
        self.preview_table.resizeColumnsToContents()
    
    def next_step(self):
        """Move to the next step."""
        if self.approve_checkbox.isChecked():
            # Store approved step
            current_step_info = self.steps[self.current_step].copy()
            current_step_info['approved'] = True
            self.approved_steps.append(current_step_info)
            
            # Move to next step
            self.current_step += 1
            
            # Update UI for next step
            self._update_ui_for_step()
        else:
            QMessageBox.warning(self, "Warning", "Please approve this step before continuing.")
    
    def finish_all_steps(self):
        """Finish all steps and emit signal."""
        if self.approve_checkbox.isChecked():
            # Store last approved step
            current_step_info = self.steps[self.current_step].copy()
            current_step_info['approved'] = True
            self.approved_steps.append(current_step_info)
            
            # Emit signal with all approved steps
            self.allStepsApproved.emit(self.approved_steps)
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please approve this step before finishing.")
    
    def reject_step(self):
        """Reject current step and continue."""
        QMessageBox.information(self, "Step Rejected", f"Step '{self.steps[self.current_step].get('name', 'Unknown')}' has been rejected and will be skipped.")
        
        # Move to next step without approval
        self.current_step += 1
        
        if self.current_step >= len(self.steps):
            # All steps processed
            self.allStepsApproved.emit(self.approved_steps)
            self.accept()
        else:
            # Update UI for next step
            self._update_ui_for_step()
    
    def cancel_workflow(self):
        """Cancel the entire workflow."""
        reply = QMessageBox.question(
            self, 
            "Cancel Workflow", 
            "Are you sure you want to cancel all steps?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.workflowCancelled.emit()
            self.reject()
    
    def _update_ui_for_step(self):
        """Update the UI for the current step."""
        if self.current_step >= len(self.steps):
            return
            
        # Update progress label
        progress_label = self.findChild(QLabel, "")
        if progress_label:
            progress_label.setText(f"Step {self.current_step + 1} of {len(self.steps)}")
        
        # Update step info
        current_step_info = self.steps[self.current_step]
        
        # Update step name
        step_name_labels = self.findChildren(QLabel)
        for label in step_name_labels:
            if "Operation:" in label.text():
                label.setText(f"Operation: {current_step_info.get('name', 'Unknown')}")
                break
        
        # Update description
        for label in step_name_labels:
            if label.text() == "Description:":
                # Find the next label which should be the description
                parent_layout = label.parent().layout()
                for i in range(parent_layout.count()):
                    item = parent_layout.itemAt(i)
                    if item.widget() == label:
                        if i + 1 < parent_layout.count():
                            desc_widget = parent_layout.itemAt(i + 1).widget()
                            if isinstance(desc_widget, QLabel):
                                desc_widget.setText(current_step_info.get('description', 'No description available'))
                        break
                break
        
        # Update code
        code_texts = self.findChildren(QTextEdit)
        for code_text in code_texts:
            code_text.setPlainText(current_step_info.get('code', 'No code available'))
            break
        
        # Update preview table
        self._update_preview_table()
        
        # Update buttons
        button_layout = self.findChild(QHBoxLayout)
        if button_layout:
            # Clear existing buttons
            for i in reversed(range(button_layout.count())):
                button_layout.itemAt(i).widget().setParent(None)
            
            # Add new buttons
            if self.current_step < len(self.steps) - 1:
                next_btn = QPushButton("Approve & Continue to Next Step")
                next_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 14px;
                        min-width: 200px;
                    }
                    QPushButton:hover {
                        background-color: #218838;
                    }
                    QPushButton:pressed {
                        background-color: #1e7e34;
                    }
                """)
                next_btn.clicked.connect(self.next_step)
                button_layout.addWidget(next_btn)
            else:
                finish_btn = QPushButton("Approve & Finish All Steps")
                finish_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 14px;
                        min-width: 200px;
                    }
                    QPushButton:hover {
                        background-color: #218838;
                    }
                    QPushButton:pressed {
                        background-color: #1e7e34;
                    }
                """)
                finish_btn.clicked.connect(self.finish_all_steps)
                button_layout.addWidget(finish_btn)
            
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
            reject_btn.clicked.connect(self.reject_step)
            
            cancel_btn = QPushButton("Cancel All")
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 14px;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
            cancel_btn.clicked.connect(self.cancel_workflow)
            
            button_layout.addWidget(reject_btn)
            button_layout.addWidget(cancel_btn)

class HelpDialog(QDialog):
    """Comprehensive help dialog explaining all functions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the help dialog UI."""
        self.setWindowTitle("Function Reference - AI Excel Assistant")
        self.setGeometry(200, 100, 1000, 800)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("AI Excel Assistant - Function Reference")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #217346;
                margin: 20px;
                text-align: center;
            }
        """)
        layout.addWidget(title_label)
        
        # Create scrollable content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
            }
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # File Operations Section
        self.add_section(content_layout, "File Operations", [
            ("Open", "Load Excel (.xlsx, .xls) or CSV files into the application. The file will be displayed in the main grid with proper column headers."),
            ("Save", "Save the current data to the previously opened file. If no file was opened, this will prompt for a new file location."),
            ("Save As", "Save the current data to a new file location. You can choose between Excel (.xlsx) and CSV formats."),
            ("Export CSV", "Export the current data to a CSV file format. Useful for sharing data with other applications."),
            ("Export Excel", "Export the current data to an Excel file format. Preserves formatting and data types.")
        ])
        
        # Data Analysis Section
        self.add_section(content_layout, "Data Analysis", [
            ("Analyze Data", "Perform comprehensive analysis of your dataset including data structure, missing values, data types, and quality issues. Provides insights and transformation suggestions."),
            ("AI Analysis", "Use natural language to ask questions about your data. The AI will understand your request and perform the appropriate Excel operations."),
            ("Test Complex Query", "Execute multi-step data operations with step-by-step approval. Each step shows a preview and requires your approval before proceeding.")
        ])
        
        # Data Processing Section
        self.add_section(content_layout, "Data Processing", [
            ("Filter", "Filter data based on specific criteria. Can be combined with other operations for complex data selection."),
            ("Sort", "Sort data by one or more columns in ascending or descending order. Useful for identifying top performers or trends."),
            ("Group By", "Group data by categories and calculate summary statistics like sum, average, count, maximum, and minimum."),
            ("Calculate", "Perform mathematical operations on numeric columns including sum, average, count, and custom formulas."),
            ("Remove Duplicates", "Identify and remove duplicate rows from your dataset to ensure data quality.")
        ])
        
        # AI Features Section
        self.add_section(content_layout, "AI Features", [
            ("Natural Language Processing", "Ask questions in plain English like 'Show me sales above $1000' or 'What is the average salary by department?'"),
            ("Automatic Code Generation", "The AI generates appropriate Python/pandas code for your requests, which you can review before execution."),
            ("Smart Suggestions", "Receive intelligent suggestions for data transformations based on your data structure and common analysis patterns."),
            ("Multi-Step Workflows", "Complex operations are broken down into manageable steps with previews and approval requirements.")
        ])
        
        # Interface Features Section
        self.add_section(content_layout, "Interface Features", [
            ("Excel-like Grid", "Familiar spreadsheet interface with proper column headers, row numbers, and grid lines for easy data viewing."),
            ("Column Headers", "Automatic detection and display of actual column names from your data files instead of generic A, B, C labels."),
            ("Data Preview", "See how your data looks at each step of processing with live previews before applying changes."),
            ("Status Updates", "Real-time feedback on operation progress and completion status in the status bar.")
        ])
        
        # Workflow Section
        self.add_section(content_layout, "Workflow Process", [
            ("1. Load Data", "Start by opening an Excel or CSV file using the File menu. The data will appear in the main grid."),
            ("2. Analyze Structure", "Use the Analyze Data function to understand your dataset and get transformation suggestions."),
            ("3. Apply Transformations", "Review and approve suggested transformations, or use AI Analysis for custom operations."),
            ("4. Preview Changes", "Each transformation shows a preview of the results before applying to your data."),
            ("5. Export Results", "Save your processed data in Excel or CSV format for further use or sharing.")
        ])
        
        # Tips Section
        self.add_section(content_layout, "Tips for Best Results", [
            ("Data Format", "Ensure your data has clear column headers in the first row for best AI analysis results."),
            ("File Size", "Large files (over 100MB) may take longer to process. Consider splitting very large datasets."),
            ("Column Names", "Use descriptive column names to help the AI understand your data structure better."),
            ("Data Types", "The application automatically detects data types, but you can review and modify them if needed."),
            ("Save Frequently", "Save your work regularly, especially after applying multiple transformations.")
        ])
        
        # Troubleshooting Section
        self.add_section(content_layout, "Troubleshooting", [
            ("File Won't Open", "Check that the file format is supported (.xlsx, .xls, .csv) and the file isn't corrupted."),
            ("AI Analysis Fails", "Ensure your question is clear and refers to columns that exist in your data."),
            ("Slow Performance", "Large datasets may take time to process. Check the status bar for progress updates."),
            ("Missing Data", "Use the Analyze Data function to identify missing values and data quality issues."),
            ("Export Errors", "Ensure you have write permissions in the target directory and sufficient disk space.")
        ])
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        close_btn.clicked.connect(self.accept)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_section(self, parent_layout, title, items):
        """Add a section with title and items to the help content."""
        # Section title
        section_title = QLabel(title)
        section_title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #495057;
                margin: 20px 0 10px 0;
                padding: 10px;
                background-color: #f8f9fa;
                border-left: 4px solid #217346;
                border-radius: 4px;
            }
        """)
        parent_layout.addWidget(section_title)
        
        # Section items
        for item_title, description in items:
            item_widget = QWidget()
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(20, 5, 20, 5)
            
            # Item title
            item_title_label = QLabel(item_title)
            item_title_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    color: #217346;
                    margin-bottom: 5px;
                }
            """)
            item_layout.addWidget(item_title_label)
            
            # Item description
            item_desc_label = QLabel(description)
            item_desc_label.setWordWrap(True)
            item_desc_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #6c757d;
                    line-height: 1.4;
                    margin-bottom: 10px;
                }
            """)
            item_layout.addWidget(item_desc_label)
            
            parent_layout.addWidget(item_widget)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #dee2e6; margin: 10px 0;")
        parent_layout.addWidget(separator)
