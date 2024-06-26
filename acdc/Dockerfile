FROM ubuntu:latest

MAINTAINER M. Ascencio-Sosa marvlad@gmail.com

RUN apt update -y
RUN apt install python3 -y
RUN apt install wget -y
RUN apt install pip -y
RUN apt install python3-numpy -y
RUN apt install python3-pandas -y
RUN apt install python3-tqdm -y
RUN apt install vim -y
RUN apt install texlive-fonts-extra -y
RUN mkdir -p /acdc
RUN cd /acdc && wget https://raw.githubusercontent.com/marvlad/Lab6Stuff/main/acdc/ACDC_WaveForm.py

EXPOSE 22
