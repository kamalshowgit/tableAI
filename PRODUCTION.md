# TableAI Production Setup Guide

## 1. Environment Variables
- Copy `.env.example` to `.env` and fill in your production values.

## 2. Dependency Management
- All dependencies are pinned in `requirements.txt`.
- Install with:
  ```bash
  pip install -r requirements.txt
  ```

## 3. Logging
- Logs are written to `tableai_app.log` in the project root.

## 4. Security
- All AI-generated code is cleaned before execution.
- For extra safety, consider running code in a container or restricted subprocess.

## 5. Caching
- Use Streamlit's `@st.cache_data` or `@st.cache_resource` for expensive operations (see code comments).

## 6. Error Handling
- Errors are logged and shown to users in a friendly way.

## 7. Testing
- Add tests for data loading, code execution, and UI flows.

## 8. Deployment
- Use a production-ready server or Docker for deployment.
- Set up CI/CD for automated testing and deployment.

## 9. Accessibility
- Test UI on multiple devices and browsers.

## 10. Documentation
- Keep this guide and code docstrings up to date.
