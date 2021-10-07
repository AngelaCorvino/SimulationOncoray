#!/usr/bin/python

import random
import subprocess
import time
import sys
import os

if (len(sys.argv) <= 2 ):
    print """\
    Usage:  job_submit.pl APPENDSTRING NEV\n
    """
    sys.exit(1)

NAME=sys.argv[1]
NEV= int(float(sys.argv[2]))

if NAME==None or not isinstance(NAME, str):
	raise Exception('invalid input: NAME should be a string')
if NEV==None or not isinstance(NEV, int):
	NEV = 1000

#FLUKA input file template:
INPUTFILE = "test_job"
#Submit-file job_submit.sh:
JFILE = "job_submit"
#job name for PBS-submission:
JNAME = "test_file"

# number of events to process:
OldSTART = 'START              1'
NewSTART = 'START     '+'{:>10.10}'.format(str(float(NEV)))

#preparation of random number seed:
OldRAND = 'RANDOMIZ         1.0'

RAND = random.randint(0,9.E8)
NewRAND = 'RANDOMIZ         1.0'+'{:>9}'.format(str(RAND))+'.'

#Open a file
try:
    with open(INPUTFILE + '.txt') as f:
                s = f.read()
    with open(INPUTFILE + '_' + NAME + '.txt', "w") as f:
                s = s.replace(OldRAND, NewRAND)
                s = s.replace(OldSTART, NewSTART)
                f.write(s)
except IOError:
                print 'cannot open or find', INPUTFILE + '.txt'

# check whether slurm exists:
slurm=os.popen('command -v sbatch').read()
if slurm != '':
    print "I use slurm!\n"
else:
    print "I use qsub\n"

try:
    if slurm != '':
	with open(JFILE + '_slurm.sh') as f:
		s = f.read()
    else:
        with open(JFILE + '.sh') as f:
                s = f.read()
    with open(JFILE + '_' + NAME + '.sh', "w") as f:
                s = s.replace(INPUTFILE, INPUTFILE + '_' + NAME)
                t = s.replace(JNAME, JNAME + '_' + NAME)
                f.write(t)
except IOError:
                print 'cannot open or find', JFILE + '.sh'
if slurm != '':
    print subprocess.check_output(['sbatch', JFILE + '_' + NAME + '.sh'])
else:
    print subprocess.check_output(['qsub', JFILE + '_' + NAME + '.sh'])
tempo=time.asctime()
print 'job_submit.pl at ' + str(tempo) + ': Finished submitting job'
