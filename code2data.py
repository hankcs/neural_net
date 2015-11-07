#!/usr/bin/python

import glob, sys, os, string

NUM_FEATURES = 20
CHUNK_SIZE = 500
NORM_FACTOR = 10.0/CHUNK_SIZE

outdict = {}

def outlayer(ext=None, exts=[]):
    if exts:
        exttup = (0,) * len(exts)
        for ndx in range(len(exts)):
            ext = exts[ndx]
            extlst = list(exttup)
            extlst[ndx] = 1
            outdict[ext] = string.replace(`extlst`[1:-1], ',','')
    elif ext:
        return outdict[ext]

def main(exts):
    outlayer(exts=exts)     # create output layer lists
    files_by_type = {}
    histogram = {}
    # Check how much source code is available, plus do histogram of chars
    for ext in exts:
        files_by_type[ext] = glob.glob('code/*.'+ext)
        total_bytes = 0
        for file in files_by_type[ext]:
            total_bytes = total_bytes + os.path.getsize(file)
            source_data = open(file).read()
            for c in source_data:
                if c not in string.letters+" \n\r\t":
                    histogram[c] = 1 + histogram.get(c,0)
        sys.stderr.write("Total bytes of %s-source: %i\n" % (ext,total_bytes))

    # Sort the historgram and get a list of input symbols
    schwartzian = []
    for char, count in histogram.items():
        schwartzian.append('%8i %s' % (count,char))
        schwartzian.sort()
        schwartzian.reverse()
    common_symbols = []
    for line in schwartzian[:NUM_FEATURES]:
        common_symbols.append(line[9])
    sys.stderr.write('Input set: '+string.join(common_symbols,' ')+'\n')

    # Create the actual data set (first randomize file order)
    allfiles = {}
    for ext, files in files_by_type.items():
        for file in files:
            allfiles[file] = ext

    # Now step through random file order
    for file, ext in allfiles.items():
        fh = open(file)
        while 1:
            chunk = fh.read(CHUNK_SIZE)
            read_len = len(chunk)
            if read_len < CHUNK_SIZE:
                break
            for c in common_symbols:
                print "%.2f" % (string.count(chunk,c)*NORM_FACTOR),
            print '>', outlayer(ext)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage:   [python] code2data.py ext [ext2 [ext3 [...]]]"
        print "Example: code2data.py c py java cpp > training.data"
    else:
        main(sys.argv[1:])
