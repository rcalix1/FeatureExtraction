import glob
i = 0
x = open("/home/seed/Desktop/TCP/Final/final.csv", 'w+') 
x.write("Class\tIP Length\tIP Header Length\tIP Flags\tTTL\tProtocol\tIP ID\tIP checksum\tSource Port\tDest Port\tSequence Number\tAck Number\tWindow Size\tTCP Header Length\tTCP flags\tTCP Length\t TCP Checksum\t TCP Stream\tTCP Urgent Pointer\n")
x.close()
for filename in glob.glob('*.csv'):
	if i < 8: #depending on the number of packets per file this number will change
		filenames=[]
		filenames.append(filename)		
		i= i+1
		with open('/home/seed/Desktop/TCP/Final/final.csv', 'a+') as outfile:
			for fname in filenames:
				with open(fname) as infile:
					for line in infile:
						outfile.write(line)
						

