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
        
RUN pip install --no-cache-dir \
    statsmodels

#fixing warning:
#Matplotlib created a temporary config/cache directory at /tmp/matplotlib-nad_6gp3 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
ENV MPLCONFIGDIR "/data/.pyplotmp/"


COPY *.py /usr/bin/

