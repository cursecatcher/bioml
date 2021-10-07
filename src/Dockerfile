FROM python:3.8-slim

RUN pip install --upgrade pip && \ 
    pip install --no-cache-dir \
        numpy \
        scipy \
        scikit-learn \
        matplotlib \
        pandas \
        seaborn \
        openpyxl 
        
COPY *.py /usr/bin/

