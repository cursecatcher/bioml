FROM python:3.8-slim

RUN pip install --upgrade pip && \ 
    pip install --no-cache-dir \
        matplotlib \
        numpy \
        scipy \
        scikit-learn \
        pandas \
        statsmodels \
        seaborn \
        openpyxl \
        kmodes 
        
# RUN pip install --no-cache-dir \
#     statsmodels

# RUN pip install --no-cache-dir kmodes 

#fixing warning:
#Matplotlib created a temporary config/cache directory at /tmp/matplotlib-nad_6gp3 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
ENV MPLCONFIGDIR "/data/.pyplotmp/"

RUN mkdir /bioml
ADD skrules /bioml/skrules  
COPY *.py /bioml/


#COPY *.py /usr/bin/

