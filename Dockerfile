FROM python:3.7-slim
COPY ./app.py /deploy/
COPY ./req.txt /deploy/
COPY ./chkpntPNN.pth /deploy/
WORKDIR /deploy/
RUN pip install -r req.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
