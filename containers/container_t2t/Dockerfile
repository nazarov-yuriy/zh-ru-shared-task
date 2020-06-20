FROM tensorflow/tensorflow:1.15.0-py3

RUN pip3 install requests tensor2tensor

COPY data /root/data
COPY run.sh /
RUN chmod a+x /run.sh

CMD ["/run.sh"]
