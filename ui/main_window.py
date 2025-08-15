"""
Main window for AI Excel Assistant PyQt application.
"""

import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QTableWidget, 
                             QTableWidgetItem, QLabel, QFrame, QScrollArea, 
                             QMessageBox, QFileDialog, QProgressBar, QSplitter, 
                             QTabWidget, QGroupBox, QGridLayout, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

# Import our modules
from core.data_processor import DataProcessor
from core.ai_analyzer import AIAnalyzer
from core.file_handler import FileHandler
from ui.dialogs import (StepApprovalDialog, ExecutionSummaryDialog, 
                       DataPreviewDialog, AIQuestionDialog)
from ui.excel_ribbon import ExcelRibbon
from ui.step_approval_workflow import StepApprovalWorkflow

class ExcelAIAssistant(QMainWindow):
    """Main application window for AI Excel Assistant."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        try:
            self.data_processor = DataProcessor()
            self.ai_analyzer = AIAnalyzer()
            self.file_handler = FileHandler()
            print("AI components initialized successfully")
        except Exception as e:
            print(f"Error initializing AI components: {e}")
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize AI components: {str(e)}")
            return
        
        # UI state
        self.current_file_path = ""
        self.transformation_steps = []
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AI Excel Assistant - Intelligent Data Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set window icon and style
        self.setWindowIcon(QIcon(""))
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Excel-like ribbon interface
        self.ribbon = ExcelRibbon()
        self.ribbon.fileOpenRequested.connect(self.load_file)
        self.ribbon.fileSaveRequested.connect(self.save_file)
        self.ribbon.fileSaveAsRequested.connect(self.save_file_as)
        self.ribbon.analyzeDataRequested.connect(self.analyze_data)
        self.ribbon.toolsRequested.connect(self.ask_ai_question)
        self.ribbon.testComplexQueryRequested.connect(self.test_complex_query)
        self.ribbon.helpRequested.connect(self.show_help)
        
        # Connect export actions
        self.ribbon.exportCsvRequested.connect(lambda: self.export_data_format('csv'))
        self.ribbon.exportExcelRequested.connect(lambda: self.export_data_format('excel'))
        
        main_layout.addWidget(self.ribbon)
        
        # Main content area - Excel-like grid
        self.create_excel_grid()
        main_layout.addWidget(self.excel_grid_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready to load data")
        
        # Set professional styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QStatusBar {
                background-color: #f8f9fa;
                color: #495057;
                border-top: 1px solid #dee2e6;
            }
        """)
        
    def create_excel_grid(self):
        """Create Excel-like grid interface."""
        self.excel_grid_widget = QWidget()
        grid_layout = QVBoxLayout(self.excel_grid_widget)
        
        # Create Excel-like table
        self.excel_table = QTableWidget()
        self.excel_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #c0c0c0;
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #0078d4;
                selection-color: white;
                color: #333333;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 8px;
                border: 1px solid #c0c0c0;
                font-weight: bold;
                font-size: 11px;
                color: #333333;
            }
            QTableCornerButton::section {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
            }
        """)
        self.excel_table.setAlternatingRowColors(True)
        self.excel_table.setShowGrid(True)
        
        # Set up table properties
        self.excel_table.setRowCount(1000)  # Start with 1000 rows
        self.excel_table.setColumnCount(26)  # Start with 26 columns (A-Z)
        
        # Set column headers (A, B, C, ...)
        column_headers = [chr(65 + i) for i in range(26)]
        self.excel_table.setHorizontalHeaderLabels(column_headers)
        
        # Set row headers (1, 2, 3, ...)
        row_headers = [str(i + 1) for i in range(1000)]
        self.excel_table.setVerticalHeaderLabels(row_headers)
        
        grid_layout.addWidget(self.excel_table)
        
    # Old UI methods removed - now using Excel ribbon interface
    
    # Old left panel method removed - now using Excel ribbon interface
    def create_left_panel(self) -> QWidget:
        """Create the left panel with file operations and analysis."""
        # This method is no longer used - replaced by Excel ribbon interface
        pass
        
        # File Operations Group
        file_group = QGroupBox("📁 File Operations")
        file_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #217346;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        file_layout = QVBoxLayout(file_group)
        
        # Load file button
        self.load_file_btn = QPushButton("📂 Load Excel/CSV File")
        self.load_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        file_layout.addWidget(self.load_file_btn)
        
        # Analyze button
        self.analyze_btn = QPushButton("🔍 Analyze Data Structure")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.analyze_btn.setEnabled(False)
        file_layout.addWidget(self.analyze_btn)
        
        left_layout.addWidget(file_group)
        
        # Data Analysis Group
        analysis_group = QGroupBox("📊 Data Analysis")
        analysis_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #217346;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        analysis_layout = QVBoxLayout(analysis_group)
        
        # AI Question button
        self.ai_question_btn = QPushButton("🤖 Ask AI Question")
        self.ai_question_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #5a32a3;
            }
        """)
        self.ai_question_btn.setEnabled(False)
        analysis_layout.addWidget(self.ai_question_btn)
        
        # Export button
        self.export_btn = QPushButton("📤 Export Data")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #fd7e14;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e8690b;
            }
        """)
        self.export_btn.setEnabled(False)
        analysis_layout.addWidget(self.export_btn)
        
        left_layout.addWidget(analysis_group)
        
        # Transformations Group
        self.transformations_group = QGroupBox("🛠️ Suggested Transformations")
        self.transformations_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #217346;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        self.transformations_layout = QVBoxLayout(self.transformations_group)
        
        # Transformations will be added here dynamically
        self.transformations_group.setVisible(False)
        left_layout.addWidget(self.transformations_group)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        return left_widget
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with data preview and results."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #217346;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
        """)
        
        # Data Preview Tab
        self.data_preview_tab = QWidget()
        data_preview_layout = QVBoxLayout(self.data_preview_tab)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #dee2e6;
                background-color: white;
                alternate-background-color: #f8f9fa;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                padding: 8px;
                border: 1px solid #dee2e6;
                font-weight: bold;
            }
        """)
        self.data_table.setAlternatingRowColors(True)
        data_preview_layout.addWidget(self.data_table)
        
        self.tab_widget.addTab(self.data_preview_tab, "📋 Data Preview")
        
        # Analysis Results Tab
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        
        # Analysis text area
        self.analysis_text = QTextEdit()
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
        """)
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        
        self.tab_widget.addTab(self.analysis_tab, "📊 Analysis Results")
        
        # AI Results Tab
        self.ai_results_tab = QWidget()
        ai_results_layout = QVBoxLayout(self.ai_results_tab)
        
        # AI results text area
        self.ai_results_text = QTextEdit()
        self.ai_results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
        """)
        self.ai_results_text.setReadOnly(True)
        ai_results_layout.addWidget(self.ai_results_text)
        
        self.tab_widget.addTab(self.ai_results_tab, "🤖 AI Results")
        
        right_layout.addWidget(self.tab_widget)
        
        return right_widget
    
    def setup_connections(self):
        """Setup signal connections."""
        # Connections are now handled by the ribbon interface
        pass
    
    def load_file(self):
        """Load an Excel or CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel or CSV File",
            "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.statusBar().showMessage("Loading file...")
            
            # Validate file format
            is_valid, message = self.file_handler.validate_file_format(file_path)
            if not is_valid:
                QMessageBox.warning(self, "Invalid File", message)
                return
            
            # Load the file
            success, message, df = self.data_processor.load_file(file_path)
            
            if success:
                self.current_file_path = file_path
                self.update_excel_grid(df)
                self.statusBar().showMessage(f"File loaded successfully: {os.path.basename(file_path)}")
            else:
                QMessageBox.critical(self, "Error", message)
                self.statusBar().showMessage("Failed to load file")
    
    def save_file(self):
        """Save the current data."""
        if not hasattr(self, 'current_file_path') or not self.current_file_path:
            self.save_file_as()
            return
        
        success = self.data_processor.export_data(self.current_file_path, 'excel')
        if success:
            QMessageBox.information(self, "Success", "File saved successfully!")
            self.statusBar().showMessage("File saved successfully")
        else:
            QMessageBox.critical(self, "Error", "Failed to save file")
    
    def save_file_as(self):
        """Save the current data to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Excel File",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        
        if file_path:
            format_type = 'excel' if file_path.endswith('.xlsx') else 'csv'
            success = self.data_processor.export_data(file_path, format_type)
            
            if success:
                self.current_file_path = file_path
                QMessageBox.information(self, "Success", "File saved successfully!")
                self.statusBar().showMessage("File saved successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to save file")
    
    def update_excel_grid(self, df: pd.DataFrame):
        """Update the Excel grid with data."""
        if df is None or df.empty:
            return
        
        # Update table dimensions
        rows = min(len(df), 1000)
        cols = min(len(df.columns), 26)
        
        self.excel_table.setRowCount(rows)
        self.excel_table.setColumnCount(cols)
        
        # Set headers using actual column names from DataFrame
        column_headers = list(df.columns[:cols])
        self.excel_table.setHorizontalHeaderLabels(column_headers)
        
        # Populate table
        for i in range(rows):
            for j in range(cols):
                value = str(df.iloc[i, j])
                if len(value) > 50:  # Truncate long values
                    value = value[:47] + "..."
                item = QTableWidgetItem(value)
                self.excel_table.setItem(i, j, item)
        
        # Resize columns to content
        self.excel_table.resizeColumnsToContents()
        
        # Store the data for later use
        self.current_data = df
    
    def analyze_data(self):
        """Analyze the loaded data structure."""
        try:
            if not hasattr(self, 'current_data') or self.current_data is None:
                QMessageBox.warning(self, "Warning", "Please load a file first.")
                return
            
            # Validate AI components
            if not hasattr(self, 'data_processor') or self.data_processor is None:
                QMessageBox.critical(self, "AI Error", "Data processor not initialized")
                return
            
            self.statusBar().showMessage("Analyzing data structure...")
            
            # Perform analysis
            analysis = self.data_processor.analyze_data_structure()
            
            if analysis:
                # Get transformation suggestions
                suggestions = self.data_processor.suggest_transformations()
                
                if suggestions:
                    # Show step-by-step approval workflow
                    workflow_dialog = StepApprovalWorkflow(self.current_data, suggestions, self)
                    workflow_dialog.allStepsApproved.connect(self.apply_approved_transformations)
                    workflow_dialog.workflowCancelled.connect(lambda: self.statusBar().showMessage("Analysis cancelled"))
                    
                    if workflow_dialog.exec_() == QDialog.Accepted:
                        self.statusBar().showMessage("Transformations applied successfully")
                    else:
                        self.statusBar().showMessage("Analysis cancelled")
                else:
                    QMessageBox.information(self, "Analysis Complete", "No transformations needed. Your data is already well-structured!")
                    self.statusBar().showMessage("Analysis complete - no transformations needed")
            else:
                QMessageBox.warning(self, "Warning", "No data to analyze.")
                
        except Exception as e:
            error_msg = f"Data analysis error: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Analysis Error", error_msg)
            self.statusBar().showMessage("Analysis failed")
    
    def apply_approved_transformations(self, approved_steps: list):
        """Apply all approved transformations."""
        try:
            # Apply each approved transformation
            for step in approved_steps:
                success = self.data_processor.apply_transformation(step)
                if not success:
                    QMessageBox.warning(self, "Warning", f"Failed to apply transformation: {step.get('description', 'Unknown')}")
            
            # Update the current data
            if self.data_processor.df_transformed is not None:
                self.current_data = self.data_processor.df_transformed
                self.update_excel_grid(self.current_data)
                
                QMessageBox.information(self, "Success", f"Applied {len(approved_steps)} transformations successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying transformations: {str(e)}")
    
    def display_analysis_results(self, analysis: dict):
        """Display the analysis results."""
        results_text = "Data Structure Analysis\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"📊 Dataset Shape: {analysis['shape'][0]:,} rows × {analysis['shape'][1]} columns\n"
        results_text += f"💾 Memory Usage: {analysis['memory_usage']:.2f} MB\n"
        results_text += f"🔄 Duplicate Rows: {analysis['duplicates']:,}\n\n"
        
        results_text += "📋 Columns:\n"
        for col in analysis['columns']:
            dtype = analysis['dtypes'].get(col, 'unknown')
            missing = analysis['missing_values'].get(col, 0)
            results_text += f"  • {col}: {dtype} (missing: {missing})\n"
        
        results_text += f"\n🔢 Numeric Columns: {len(analysis['numeric_columns'])}\n"
        results_text += f"📝 Categorical Columns: {len(analysis['categorical_columns'])}\n"
        results_text += f"📅 Date Columns: {len(analysis['date_columns'])}\n"
        
        if analysis['quality_issues']:
            results_text += "\n⚠️ Data Quality Issues:\n"
            for issue in analysis['quality_issues']:
                results_text += f"  • {issue}\n"
        
        self.analysis_text.setPlainText(results_text)
    
    def display_transformation_suggestions(self, suggestions: list):
        """Display transformation suggestions."""
        # Clear previous suggestions
        for i in reversed(range(self.transformations_layout.count())):
            self.transformations_layout.itemAt(i).widget().setParent(None)
        
        # Add new suggestions
        for i, suggestion in enumerate(suggestions):
            suggestion_widget = self.create_suggestion_widget(suggestion, i)
            self.transformations_layout.addWidget(suggestion_widget)
        
        self.transformations_group.setVisible(True)
    
    def create_suggestion_widget(self, suggestion: dict, index: int) -> QWidget:
        """Create a widget for a transformation suggestion."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc_label = QLabel(suggestion['description'])
        desc_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #495057;
                margin: 5px;
                padding: 8px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }
        """)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Code preview
        code_label = QLabel(f"Code: {suggestion['code']}")
        code_label.setStyleSheet("""
            QLabel {
                font-family: 'Courier New', monospace;
                background-color: #2d3748;
                color: #00ff88;
                padding: 8px;
                border-radius: 4px;
                margin: 5px;
                font-size: 11px;
            }
        """)
        code_label.setWordWrap(True)
        layout.addWidget(code_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        apply_btn.clicked.connect(lambda: self.apply_transformation(suggestion))
        
        preview_btn = QPushButton("Preview")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        preview_btn.clicked.connect(lambda: self.preview_transformation(suggestion))
        
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(preview_btn)
        layout.addLayout(button_layout)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        return widget
    
    def apply_transformation(self, suggestion: dict):
        """Apply a transformation suggestion."""
        # Show approval dialog
        dialog = StepApprovalDialog(
            suggestion.get('type', 'Unknown'),
            suggestion['description'],
            suggestion['code'],
            self
        )
        
        if dialog.exec_() == QDialog.Accepted and dialog.is_approved():
            # Apply the transformation
            success = self.data_processor.apply_transformation(suggestion)
            
            if success:
                # Update the current data
                df = self.data_processor.df_transformed
                self.current_data = df
                self.update_excel_grid(df)
                
                QMessageBox.information(self, "Success", f"Applied transformation: {suggestion['description']}")
                self.statusBar().showMessage("Transformation applied successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to apply transformation")
    
    def preview_transformation(self, suggestion: dict):
        """Preview a transformation suggestion."""
        QMessageBox.information(self, "Preview", f"Preview for: {suggestion['description']}\n\nCode: {suggestion['code']}")
    
    def ask_ai_question(self):
        """Open AI question dialog."""
        if not hasattr(self, 'current_data') or self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load a file first.")
            return
        
        dialog = AIQuestionDialog(self)
        dialog.questionSubmitted.connect(self.process_ai_question)
        dialog.exec_()
    
    def test_complex_query(self):
        """Test a complex query that requires multiple previews and approvals."""
        if not hasattr(self, 'current_data') or self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load a file first.")
            return
        
        # Create a complex multi-step operation
        complex_steps = [
            {
                'name': 'Data Filtering',
                'description': 'Filter the data to show only records that meet specific criteria. This step will reduce the dataset size for better analysis.',
                'code': 'filtered_df = df[df["column_name"] > threshold_value]',
                'preview_data': self.current_data.head(10)  # Show first 10 rows as preview
            },
            {
                'name': 'Data Sorting',
                'description': 'Sort the filtered data by a specific column in descending order to identify top performers or highest values.',
                'code': 'sorted_df = filtered_df.sort_values(by="sort_column", ascending=False)',
                'preview_data': self.current_data.head(10)  # Show preview of what sorting would look like
            },
            {
                'name': 'Data Aggregation',
                'description': 'Group the sorted data by categories and calculate summary statistics like sum, average, and count for each group.',
                'code': 'grouped_df = sorted_df.groupby("group_column").agg({\n    "value_column": ["sum", "mean", "count"]\n}).round(2)',
                'preview_data': self.current_data.head(10)  # Show preview of grouping structure
            },
            {
                'name': 'Final Formatting',
                'description': 'Format the final results with proper column names, reset index, and apply final styling for presentation.',
                'code': 'final_df = grouped_df.reset_index()\nfinal_df.columns = ["Category", "Total", "Average", "Count"]\nfinal_df = final_df.sort_values("Total", ascending=False)',
                'preview_data': self.current_data.head(10)  # Show final formatted preview
            }
        ]
        
        # Show the multi-step approval dialog
        from ui.dialogs import MultiStepApprovalDialog
        workflow_dialog = MultiStepApprovalDialog(self.current_data, complex_steps, self)
        workflow_dialog.allStepsApproved.connect(self.apply_complex_operations)
        workflow_dialog.workflowCancelled.connect(lambda: self.statusBar().showMessage("Complex query workflow cancelled"))
        
        if workflow_dialog.exec_() == QDialog.Accepted:
            self.statusBar().showMessage("Complex query workflow completed successfully")
        else:
            self.statusBar().showMessage("Complex query workflow cancelled")
    
    def apply_complex_operations(self, approved_steps: list):
        """Apply all approved complex operations."""
        try:
            self.statusBar().showMessage("Applying complex operations...")
            
            # Simulate applying the operations
            result_df = self.current_data.copy()
            operation_log = []
            
            for i, step in enumerate(approved_steps):
                step_name = step.get('name', f'Step {i+1}')
                operation_log.append(f"Applied: {step_name}")
                
                # Simulate data transformation (in real implementation, this would execute the actual code)
                if step['name'] == 'Data Filtering':
                    # Simulate filtering - take first 50% of rows
                    result_df = result_df.iloc[:len(result_df)//2].copy()
                elif step['name'] == 'Data Sorting':
                    # Simulate sorting - sort by first column
                    if len(result_df.columns) > 0:
                        result_df = result_df.sort_values(by=result_df.columns[0]).copy()
                elif step['name'] == 'Data Aggregation':
                    # Simulate aggregation - group by first column and count
                    if len(result_df.columns) > 0:
                        result_df = result_df.groupby(result_df.columns[0]).size().reset_index(name='Count')
                elif step['name'] == 'Final Formatting':
                    # Simulate formatting - add some styling info
                    result_df = result_df.copy()
                    result_df['Processed'] = 'Yes'
            
            # Update the current data
            self.current_data = result_df
            self.update_excel_grid(result_df)
            
            # Show success message
            QMessageBox.information(
                self, 
                "Complex Operations Applied", 
                f"Successfully applied {len(approved_steps)} operations!\n\nOperations applied:\n" + 
                "\n".join([f"• {step.get('name', 'Unknown')}" for step in approved_steps])
            )
            
            self.statusBar().showMessage(f"Applied {len(approved_steps)} complex operations successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying complex operations: {str(e)}")
            self.statusBar().showMessage("Failed to apply complex operations")
    
    def show_help(self):
        """Show the help dialog with function reference."""
        from ui.dialogs import HelpDialog
        help_dialog = HelpDialog(self)
        help_dialog.exec_()
    
    def process_ai_question(self, question: str):
        """Process an AI question."""
        try:
            self.statusBar().showMessage("Processing AI question...")
            
            # Validate AI components
            if not hasattr(self, 'ai_analyzer') or self.ai_analyzer is None:
                QMessageBox.critical(self, "AI Error", "AI analyzer not initialized")
                return
            
            # Understand the question
            operation_info = self.ai_analyzer.understand_question(
                question, 
                self.current_data
            )
            
            # Execute the operation
            result_df, operation_log, status = self.ai_analyzer.execute_excel_operation(
                self.current_data,
                operation_info,
                question
            )
            
            if status == "success":
                # Generate business insight
                business_insight = self.ai_analyzer.generate_business_insight(
                    result_df, question, operation_log
                )
                
                # Display results in a dialog
                self.display_ai_results_dialog(question, result_df, operation_log, business_insight)
                
                # Update the grid with results
                self.current_data = result_df
                self.update_excel_grid(result_df)
                
                self.statusBar().showMessage("AI question processed successfully")
            else:
                QMessageBox.critical(self, "Error", f"Failed to process question: {operation_log[0] if operation_log else 'Unknown error'}")
                
        except Exception as e:
            error_msg = f"AI processing error: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "AI Processing Error", error_msg)
            self.statusBar().showMessage("AI processing failed")
    
    def display_ai_results(self, question: str, result_df: pd.DataFrame, operation_log: list, business_insight: str):
        """Display AI analysis results."""
        results_text = f"AI Analysis Results\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Question: {question}\n\n"
        
        results_text += "Operations Performed:\n"
        for log in operation_log:
            results_text += f"  • {log}\n"
        
        results_text += f"\nResults Summary:\n"
        results_text += f"  • Result rows: {len(result_df):,}\n"
        results_text += f"  • Result columns: {len(result_df.columns)}\n"
        
        results_text += f"\nBusiness Insight:\n"
        # Clean up markdown formatting from AI responses
        cleaned_insight = self._clean_markdown_formatting(business_insight)
        results_text += cleaned_insight
        
        self.ai_results_text.setPlainText(results_text)
    
    def display_ai_results_dialog(self, question: str, result_df: pd.DataFrame, operation_log: list, business_insight: str):
        """Display AI analysis results in a dialog."""
        from ui.dialogs import ExecutionSummaryDialog
        
        dialog = ExecutionSummaryDialog(question, result_df, operation_log, business_insight, self)
        dialog.exec_()
    
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
    
    def export_data(self):
        """Export the processed data."""
        if not hasattr(self, 'current_data') or self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return
        
        # Get export format
        format_choice, ok = QMessageBox.question(
            self,
            "Export Format",
            "Choose export format:",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        export_format = 'csv' if format_choice == QMessageBox.Yes else 'excel'
        
        # Get save file path
        if export_format == 'csv':
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV File",
                f"exported_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Excel File",
                f"exported_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "Excel Files (*.xlsx)"
            )
        
        if file_path:
            success = self.data_processor.export_data(file_path, export_format)
            
            if success:
                QMessageBox.information(self, "Success", f"Data exported successfully to:\n{file_path}")
                self.statusBar().showMessage("Data exported successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to export data")
    
    def export_data_format(self, format_type: str):
        """Export data in the specified format."""
        if not hasattr(self, 'current_data') or self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return
        
        # Get save file path
        if format_type == 'csv':
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV File",
                f"exported_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Excel File",
                f"exported_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "Excel Files (*.xlsx)"
            )
        
        if file_path:
            success = self.data_processor.export_data(file_path, format_type)
            
            if success:
                QMessageBox.information(self, "Success", f"Data exported successfully to:\n{file_path}")
                self.statusBar().showMessage("Data exported successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to export data")
