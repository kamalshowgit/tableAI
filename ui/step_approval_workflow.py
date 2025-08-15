"""
Step-by-step approval workflow for data transformations.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QLabel, QPushButton, QProgressBar, QFrame,
                             QScrollArea, QWidget, QGridLayout, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import pandas as pd

class StepApprovalWorkflow(QDialog):
    """Dialog for step-by-step approval workflow."""
    
    # Signals
    allStepsApproved = pyqtSignal(list)  # List of approved steps
    workflowCancelled = pyqtSignal()
    
    def __init__(self, original_data: pd.DataFrame, transformation_steps: list, parent=None):
        super().__init__(parent)
        self.original_data = original_data
        self.transformation_steps = transformation_steps
        self.approved_steps = []
        self.current_step_index = 0
        
        self.init_ui()
        self.setup_workflow()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Step-by-Step Data Transformation Approval")
        self.setGeometry(100, 100, 1200, 800)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Data Transformation Workflow")
        header_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #217346;
                margin: 10px;
                text-align: center;
            }
        """)
        layout.addWidget(header_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #c0c0c0;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #217346;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Step information
        step_info_layout = QHBoxLayout()
        
        # Left side - Step details
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.step_title_label = QLabel("Step Information")
        self.step_title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
                margin: 5px;
            }
        """)
        left_layout.addWidget(self.step_title_label)
        
        self.step_description_label = QLabel("Description will appear here")
        self.step_description_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #333333;
                margin: 5px;
                padding: 10px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                font-weight: normal;
            }
        """)
        self.step_description_label.setWordWrap(True)
        left_layout.addWidget(self.step_description_label)
        
        # Code preview
        code_label = QLabel("Generated Code:")
        code_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(code_label)
        
        self.code_text = QTextEdit()
        self.code_text.setMaximumHeight(100)
        self.code_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d3748;
                color: #00ff88;
                border: 1px solid #4a5568;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 10px;
            }
        """)
        self.code_text.setReadOnly(True)
        left_layout.addWidget(self.code_text)
        
        # Approval checkbox
        self.approve_checkbox = QCheckBox("Approve this transformation step")
        self.approve_checkbox.setChecked(True)
        self.approve_checkbox.setStyleSheet("""
            QCheckBox {
                margin: 15px 0;
                font-weight: bold;
                font-size: 14px;
                color: #217346;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        left_layout.addWidget(self.approve_checkbox)
        
        step_info_layout.addWidget(left_panel)
        
        # Right side - Data preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        preview_label = QLabel("Data Preview (After This Step)")
        preview_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
                margin: 5px;
            }
        """)
        right_layout.addWidget(preview_label)
        
        # Data preview area
        self.data_preview_text = QTextEdit()
        self.data_preview_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 10px;
                color: #333333;
            }
        """)
        self.data_preview_text.setReadOnly(True)
        right_layout.addWidget(self.data_preview_text)
        
        step_info_layout.addWidget(right_panel)
        
        layout.addLayout(step_info_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:disabled {
                background-color: #adb5bd;
            }
        """)
        self.prev_btn.clicked.connect(self.previous_step)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.next_btn.clicked.connect(self.next_step)
        
        self.finish_btn = QPushButton("Finish & Apply All")
        self.finish_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.finish_btn.clicked.connect(self.finish_workflow)
        self.finish_btn.setVisible(False)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel_workflow)
        
        button_layout.addWidget(self.prev_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.finish_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
    def setup_workflow(self):
        """Setup the workflow with transformation steps."""
        if not self.transformation_steps:
            self.step_title_label.setText("No transformations available")
            self.next_btn.setEnabled(False)
            return
            
        self.total_steps = len(self.transformation_steps)
        self.progress_bar.setMaximum(self.total_steps)
        self.progress_bar.setValue(0)
        
        self.show_current_step()
        
    def show_current_step(self):
        """Show the current step information."""
        if self.current_step_index >= len(self.transformation_steps):
            return
            
        step = self.transformation_steps[self.current_step_index]
        
        # Update step information
        self.step_title_label.setText(f"Step {self.current_step_index + 1} of {self.total_steps}")
        self.step_description_label.setText(step.get('description', 'No description available'))
        self.code_text.setPlainText(step.get('code', 'No code available'))
        
        # Update progress
        self.progress_bar.setValue(self.current_step_index + 1)
        
        # Update button states
        self.prev_btn.setEnabled(self.current_step_index > 0)
        
        if self.current_step_index == len(self.transformation_steps) - 1:
            self.next_btn.setVisible(False)
            self.finish_btn.setVisible(True)
        else:
            self.next_btn.setVisible(True)
            self.finish_btn.setVisible(False)
        
        # Show data preview for this step
        self.update_data_preview()
        
    def update_data_preview(self):
        """Update the data preview for the current step."""
        try:
            # Simulate the transformation to show preview
            current_df = self.original_data.copy()
            
            # Apply all steps up to the current one
            for i in range(self.current_step_index + 1):
                step = self.transformation_steps[i]
                if self.approve_checkbox.isChecked():
                    # Apply the transformation
                    current_df = self.apply_transformation_preview(current_df, step)
            
            # Show preview
            preview_text = f"Data Preview after Step {self.current_step_index + 1}:\n"
            preview_text += f"Shape: {current_df.shape[0]} rows × {current_df.shape[1]} columns\n\n"
            preview_text += "First 10 rows:\n"
            preview_text += current_df.head(10).to_string()
            
            self.data_preview_text.setPlainText(preview_text)
            
        except Exception as e:
            self.data_preview_text.setPlainText(f"Error generating preview: {str(e)}")
    
    def apply_transformation_preview(self, df: pd.DataFrame, step: dict) -> pd.DataFrame:
        """Apply a transformation for preview purposes."""
        try:
            # This is a simplified preview - in real implementation, you'd use the actual transformation logic
            if step.get('type') == 'remove_duplicates':
                return df.drop_duplicates()
            elif step.get('type') == 'fill_missing_numeric':
                col = step.get('column')
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            # Add more transformation types as needed
            
            return df
        except Exception:
            return df
    
    def next_step(self):
        """Move to the next step."""
        if self.current_step_index < len(self.transformation_steps) - 1:
            # Save approval status for current step
            self.save_step_approval()
            
            self.current_step_index += 1
            self.show_current_step()
    
    def previous_step(self):
        """Move to the previous step."""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.show_current_step()
    
    def save_step_approval(self):
        """Save the approval status for the current step."""
        if self.current_step_index < len(self.transformation_steps):
            step = self.transformation_steps[self.current_step_index]
            step['approved'] = self.approve_checkbox.isChecked()
            
            if step['approved']:
                self.approved_steps.append(step)
    
    def finish_workflow(self):
        """Finish the workflow and apply all approved steps."""
        # Save approval for current step
        self.save_step_approval()
        
        # Filter only approved steps
        final_approved_steps = [step for step in self.transformation_steps if step.get('approved', False)]
        
        if final_approved_steps:
            self.allStepsApproved.emit(final_approved_steps)
            self.accept()
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Steps Approved", "No transformation steps were approved.")
    
    def cancel_workflow(self):
        """Cancel the workflow."""
        self.workflowCancelled.emit()
        self.reject()
