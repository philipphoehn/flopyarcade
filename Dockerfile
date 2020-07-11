FROM python:3.7
RUN mkdir -p /home/FloPyArcade
COPY . /home/FloPyArcade
EXPOSE 81
