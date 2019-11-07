#############################################################################################

from scapy.all import *
import os
import time
import re
import zlib
from mimetools import Message
import npyscreen
import subprocess
import json
from collections import OrderedDict

#############################################################################################

contentType = ['multipart/form-data','text','video','audio', 'image']

i=0
v = ""

captureDurationTypeDict =[' -c ',
                          ' -a files: ' ,
                          ' -a duration: ',
                          " -a filesize: "]

captureFieldsDict = {
0: ' -e _ws.col.Info ',
    1:' -e http ' ,
    2: ' -e frame.number ',
    3:" -e ip.addr ",

}

outputType = {
0: 'text',
1:'forms' ,
2: 'image',
3:" audio",

}

############################################################################################

# os.system("tshark  -T fields -e _ws.col.Info -e http -e frame.time -e  "
# "data.data -w Eavesdrop_Data.pcap -c 1000")

x = "tshark  -T fields -e _ws.col.Info -e http -e frame.time -e data.data -w E.pcap -c 1"
y = 'http.pcap'


###########################################################################################

def eavesdrop(x,y,T):
    subprocess.call(x, shell=True)
    if os.path.isfile(y):
        data = y
        a = rdpcap(data) ## scapy function creates data structure of packet
        sessions = a.sessions()
        text_file = open("Output.txt", "w")
        for session in sessions:
            http_payload = ""
            for packet in sessions[session]:
                try:
                    if packet[TCP].dport == 80 or packet[TCP].sport == 80:
                        http_payload += str(packet[TCP].payload)
                except:
                    pass

            headers = HTTPHeaders(http_payload)

            if headers is None:
                continue
            text = extractText(headers,http_payload,T)

            if text is not None:
                try:

                    text_file.write("Payload::  " + '\n' + text + '\n')

                except:
                    text_file.write("Something went wrong + \n")
        text_file.close()




###########################################################################################


    def contSniff(self):
        count = 0
        data = ""
        p = subprocess.Popen("tshark -V  -l -p  -S '::::END OF PACKET::::::' ", stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=True)
                             for line in iter(p.stdout.readline, '\n\r\n'):
                                 if ('::::END OF PACKET::::::' not in line):
                                     data += line
                                         else:
                                             packet = data
                                                 data = ""
                                                     print(self.parsePacket(packet))
                                                         if "malformed" in data:
                                                             count += 1



############################################################################################


def HTTPHeaders(http_payload):
    try:
        # isolate headers
        headers_raw = http_payload[:http_payload.index("\r\n\r\n") + 2]
     
        regex = ur"(?:[\r\n]{0,1})(\w+\-\w+|\w+)(?:\ *:\ *)([^\r\n]*)(?:[\r\n]{0,1})"
        headers = dict(re.findall(regex, headers_raw))
        print headers
        return headers
    except:
        return None
    if 'Content-Type' not in headers:
        return None
    return headers


#############################################################################################


def extractText(headers, http_payload, type):
        text = None
        try:
            if type in headers['Content-Type']:
                text = http_payload[http_payload.index("\r\n\r\n")+3:]
                try:
                    if "Accept-Encoding" in headers.keys():
                        if headers['Accept-Encoding'] == "gzip":
                            text = zlib.decompress(text,  16+zlib.MAX_WBITS)
                    elif headers['Content-Encoding'] == "deflate":
                        text = zlib.decompress(text)
                except: pass
        except:
            return None
        return text


########################################################################################

print("<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>")
