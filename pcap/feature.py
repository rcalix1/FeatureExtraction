import os
import glob
i = 1	#counter to name temp file
j = 1	#counter to name final file
r = 1	#counter to delete temp file
string_add = 0 #class number (change this depending on the class 0=Assistant 1=Camera 2=Miscellaneous 3=Moble 4=Outlet)

#creates the first file in order to put the header in and everything after this is appended into a file


#goes through the current directory and looks for every file with .pcap
for filename in glob.glob('*.pcap'):
	#puts tshark output into a temp file
	variable = str(os.system("tshark -r " + filename + " -T fields -e ip.len -e ip.hdr_len -e ip.flags -e ip.ttl -e ip.proto -e ip.id -e ip.checksum -e tcp.srcport -e tcp.dstport -e tcp.seq -e tcp.ack -e tcp.window_size_value -e tcp.hdr_len -e tcp.flags -e tcp.len -e tcp.checksum -e tcp.stream -e tcp.urgent_pointer >> zzz" + str(i)+ ".csv"))
	
	g = open("zzz"+str(i)+".csv", 'a+')	
	g.write(variable)
	g.close

	
	#opens the temp file and the final file
	l = open("zzz"+str(i)+".csv", 'r')
	final = open("AssistantTCP"+str(j)+".csv", 'a+')
	
	#goes through the final file and adds the class number at the beginning of every line	
	for line in l:
		final.write(str(string_add)+ '\t' + line.rstrip() + '\n')
	final.close()

	#removes the temporary files	
	os.system("rm zzz"+str(r)+".csv")	
	j = j+1
	i = i+1
	r = r+1
	
