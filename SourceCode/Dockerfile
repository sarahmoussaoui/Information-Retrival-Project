# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Set environment variables
ENV PORT=8501

# Run Streamlit
CMD streamlit run src/ui/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
