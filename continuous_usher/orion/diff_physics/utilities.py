import os,subprocess

def create_directory(directory):
    try:
        os.mkdir(directory)
    except OSError:
        print ("Creation of the directory %s failed"%directory)
    else:
        print ("Successfully created the directory %s"%directory)

def execute_command(command):
    p=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE).communicate()

def write_to_text_file(filename,value):
    with open(filename,'w') as f:
        f.write('%d'%value)
