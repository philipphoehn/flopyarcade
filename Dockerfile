FROM python:3.7-slim

# installing temporarily required system programs
RUN apt-get update -y
RUN apt-get install -y gfortran git

# cloning FloPyArcade repository and installing its dependencies
RUN mkdir -p /FloPyArcade
WORKDIR /FloPyArcade
RUN git clone -b master https://github.com/philipphoehn/FloPyArcade.git .
RUN python3 -m pip install -r /FloPyArcade/requirements.txt

# installing simulators
RUN mkdir -p /FloPyArcade/simulators
RUN python3 -m pip install requests
WORKDIR /usr/local/lib/python3.7/site-packages/pymake
RUN git clone https://github.com/modflowpy/pymake.git .
RUN python3 setup.py install
WORKDIR /FloPyArcade/simulators
RUN python3 /usr/local/lib/python3.7/site-packages/pymake/examples/make_mf2005.py
RUN python3 /usr/local/lib/python3.7/site-packages/pymake/examples/make_mp6.py
WORKDIR /FloPyArcade

# removing spamming print state in FloPy (https://wiki.ubuntuusers.de/sed/)
# RUN python3 -c "import subprocess; subprocess.call(['sed', '-i', '/.*write(line).*/write(str())', '/usr/local/lib/python3.7/site-packages/flopy/mbase.py'])"

# removing temporarily required system programs
RUN apt-get remove -y git
RUN apt autoremove -y

EXPOSE 81
