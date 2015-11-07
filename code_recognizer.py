#!/usr/bin/python

import string, sys
from bpnn import *

TRAINSIZE = 500
TESTSIZE = 500
HIDDEN = 8
ITERATIONS = 1000
LEARNRATE = 0.5
MOMENTUM = 0.1
ERROR_CUTOFF = 0.01

class NN2(NN):
    def train(self, patterns, iterations=2000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for var, val in globals().items():
            if var == string.upper(var) and len(var) > 3:
                sys.stderr.write('%s = %s\n' % (var,val))
        sys.stderr.write('error -> ')
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.runNN(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 10 == 0:
                sys.stderr.write('%1.3f... ' % error)
            # No need to keep iterating if error is very low
            if error < ERROR_CUTOFF:
                break

    def test(self, patterns):
        hits = 0
        for p in patterns:
            hits = hits + self.compare(p[1], self.runNN(p[0]))
        hit_rate = (100.0*hits)/len(patterns)
        sys.stderr.write('\nSuccess rate against test data: %.2f%%\n' % hit_rate)

    def compare(self, targets, activations):
        matches = 0
        print "Targets:", targets, "-- Errors:",
        for n in range(len(targets)):
            error = abs(targets[n]-activations[n])
            print '(%.3f' % error,
            if error < 0.5:
                print 'OK)  ',
                matches = matches+1
            else:
                print 'FAIL)',
        if matches == self.no:
            print '-- SUCCESS!'
            return 1
        else:
            print '-- FAILURE!'
            return 0

# Override constants if corresponding switches given
# naive parser is -really- easy to crash w/ funny switches: deal with it
def parse_switches(switches):
    for switch in switches:
        if switch[0] in ('-','/'):  # it's a switch (ignore otherwise)
            var, val = string.split(switch[1:],'=')
            globals()[var] = eval(val)


if __name__ == '__main__':
    parse_switches(sys.argv[2:])        # Check for any non-default options

    # Open the input data, and read a sample line for parameters
    fh_data = open(sys.argv[1], 'r')
    line = fh_data.readline()
    inputs, targets = string.split(line,'>')

    # create a network (input, hidden, output)
    INPUTS = len(string.split(inputs))
    OUTPUTS = len(string.split(targets))
    net = NN2(INPUTS, HIDDEN, OUTPUTS)

    # create the training and testing data
    trainpat = []
    testpat = []
    for n in xrange(TRAINSIZE+TESTSIZE):
        line = fh_data.readline()
        try:
            inputs, targets = string.split(line,'>')
        except:
            raise IOError, 'Input data set is not large enough'
        in_lst = string.split(inputs)
        trg_lst = string.split(targets)

        for i in range(len(in_lst)):        # Convert strings to numbers
            in_lst[i] = eval(in_lst[i])
        for i in range(len(trg_lst)):
            trg_lst[i] = eval(trg_lst[i])

        if n < TRAINSIZE:
            trainpat.append([in_lst,trg_lst])
        else:
            testpat.append([in_lst,trg_lst])

    # Eat the rest of the input, in case piped in
    fh_data.read()

    # train it with some patterns
    net.train(trainpat, iterations=ITERATIONS, N=LEARNRATE, M=MOMENTUM)

    # test it
    net.test(testpat)
    #net.weights()

