# 🤖 AI Excel Assistant

A powerful, intelligent Excel data analysis and transformation tool built with PyQt5.

## ✨ Features

- **📁 File Support**: Load Excel (.xlsx, .xls) and CSV files
- **🔍 Smart Analysis**: Automatic data structure analysis and quality assessment
- **🛠️ AI-Powered Transformations**: Intelligent suggestions for data cleaning and transformation
- **🤖 Natural Language Processing**: Ask questions about your data in plain English
- **📊 Interactive Data Preview**: View and explore your data in a user-friendly table
- **📤 Export Options**: Save processed data in multiple formats
- **🎨 Modern UI**: Clean, professional interface with intuitive controls

## 🏗️ Architecture

The application is built with a modular, clean architecture:

```
ai_excel_assistant/
├── main.py                 # Main application entry point
├── ui/                     # User interface components
│   ├── main_window.py      # Main application window
│   └── dialogs.py          # Dialog boxes and popups
├── core/                   # Core business logic
│   ├── data_processor.py   # Data loading and processing
│   ├── ai_analyzer.py      # AI analysis and natural language processing
│   └── file_handler.py     # File operations and utilities
├── utils/                  # Utility functions
│   └── helpers.py          # Helper functions and utilities
└── requirements.txt        # Python dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- PyQt5

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd ai_excel_assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Running the Application
```bash
python main.py
```

### Basic Workflow
1. **Load Data**: Click "📂 Load Excel/CSV File" to select your data file
2. **Analyze**: Click "🔍 Analyze Data Structure" to get insights about your data
3. **Transform**: Review and apply suggested transformations
4. **Ask AI**: Use "🤖 Ask AI Question" to query your data in natural language
5. **Export**: Save your processed data using "📤 Export Data"

### AI Question Examples
- "Show me sales above $1000"
- "What is the average salary?"
- "Group by department and count employees"
- "Find all customers from New York"
- "Compare sales vs expenses"
- "Show trend analysis over time"

## 🔧 Development

### Project Structure
- **`main.py`**: Application entry point and main window creation
- **`ui/`**: All PyQt user interface components
- **`core/`**: Business logic and data processing
- **`utils/`**: Helper functions and utilities

### Adding New Features
1. **UI Components**: Add new dialogs in `ui/dialogs.py`
2. **Data Processing**: Extend functionality in `core/data_processor.py`
3. **AI Analysis**: Enhance natural language processing in `core/ai_analyzer.py`
4. **Utilities**: Add helper functions in `utils/helpers.py`

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Add docstrings for all functions and classes
- Keep functions focused and single-purpose

## 📋 Requirements

- **PyQt5**: GUI framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **openpyxl**: Excel file reading/writing

## 🐛 Troubleshooting

### Common Issues
1. **PyQt5 Import Error**: Ensure PyQt5 is properly installed
2. **File Loading Issues**: Check file format and permissions
3. **Memory Issues**: Large files may require more RAM

### Logs
Application logs are saved to `ai_excel_assistant.log` for debugging.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyQt5 for the user interface
- Powered by pandas and numpy for data processing
- Inspired by modern data analysis tools

---

**Made with ❤️ by Local AI**
