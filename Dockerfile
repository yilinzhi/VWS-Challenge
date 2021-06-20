# Use clatfd/vw:version1 as a parent image
#FROM clatfd/vw:version1
#
#ENV PATH="/root/miniconda3/bin:${PATH}"
#ARG PATH="/root/miniconda3/bin:${PATH}"
#
#RUN mkdir /VW
#
#RUN apt-get update \
#    && apt-get install -y libxml2-dev libxslt-dev python-dev zlib1g-dev
#
## Set the working directory
#WORKDIR /
#
## Copy the current directory contents into the container
#COPY . /VW
#
#
#RUN wget \
#    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#    && mkdir /root/.conda \
#    && bash Miniconda3-latest-Linux-x86_64.sh -b \
#    && rm -f Miniconda3-latest-Linux-x86_64.sh
#RUN conda --version

# Install any needed packages specified in requirements.txt

FROM yilinzhi/vw:team4

# Set the working directory
WORKDIR /



















