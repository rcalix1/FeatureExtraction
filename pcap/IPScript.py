import os
import glob
i = 1
for filename in glob.glob('*.pcap'):
    os.system("tshark -r " + filename + " -w /home/seed/Desktop/UDP/Mobile/UDPmobie" + str(i) +  ".pcap -Y 'udp && (ip.src==192.168.1.45)'")
    i = i +1
