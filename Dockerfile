FROM python:3.11.1

WORKDIR /ecomchurn

ENV HOST 0.0.0.0

COPY . /ecomchurn
EXPOSE 8501
RUN pip install -r requirements.txt
CMD streamlit run server.py
