FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Streamlit config
RUN mkdir -p ~/.streamlit && echo "\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
[theme]\n\
base = 'dark'\n\
primaryColor = '#e84545'\n\
backgroundColor = '#0e1117'\n\
secondaryBackgroundColor = '#1a1f2e'\n\
textColor = '#e0e0e0'\n\
" > ~/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
