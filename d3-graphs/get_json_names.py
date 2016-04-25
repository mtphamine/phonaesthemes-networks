import os

outfile = open('jsonNames.txt', 'w')

os.chdir('json_files')
fileNames = os.listdir(os.getcwd())

for name in fileNames:
	splitName = name.split('_')
	print splitName
	outfile.write('<OPTION VALUE="json_files/'+name+'">\n'+splitName[0]+'\n')

outfile.close()