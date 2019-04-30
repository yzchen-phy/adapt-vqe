#!/usr/bin/env python
import argparse,re,math,os
import csv
import numpy as np
import mmap

#
#   Setup input arguments
parser = argparse.ArgumentParser(description='Take a list of text files and a string to match and create a csv')
parser.add_argument('files', nargs='+', help='List of text files to process')


re_energy = re.compile('Finished:\s*(\S*)')
re_norm = re.compile('Norm of \S* =\s*(\S*)')
re_max = re.compile('Max  of \S* =\s*(\S*)')
re_ref = re.compile('State\s*0:\s*(\S*)')
#re_ref = re.compile('Reference Energy:\s*(\S*)')


args = vars(parser.parse_args())
files = args['files']

energies = []
norms = []
maxs = []
errs = []

print("Process files:  ", files)
for filename in files:
    f = open(filename, "r")
    f_energy = re_energy.findall(f.read())
    f.seek(0)
    f_norm   = re_norm.findall(f.read())
    f.seek(0)
    f_max    = re_max.findall(f.read())
    f.seek(0)
    f_ref    = re_ref.findall(f.read())
    
    f_max =  [np.abs(float(m)) for m in f_max]

    energies.append(f_energy)
    norms.append(f_norm)
    maxs.append(f_max)
    
    assert(len(f_ref)==1)
    err = [(float(x)- float(f_ref[0]))*627.51 for x in f_energy]
    errs.append(err)


with open("energies.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(energies)

with open("norm.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(norms)

with open("max.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(maxs)

with open("errs.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(errs)
