import subprocess, time, sys, getopt, os

MaxPipes = 30
NRUN = 1000
stagger = 0.1

output = None  #open("/dev/null", "w")

opts, args = getopt.getopt(sys.argv[1:], "m:n:s:N:")
for opt, val in opts:
    if opt == '-m': MaxPipes = int(val)
    if opt == '-n': NRUN = int(val)
    if opt == '-s': stagger = float(val)
    if opt == '-N': 
        NRUN = int(val)
        MaxPipes = NRUN

command = args

Pipes = []

ndone = 0
nstarted = 0
tlast = 0

t0 = time.time()

while nstarted < NRUN:
    while len(Pipes) < MaxPipes and nstarted < NRUN:
        now = time.time()
        if now < tlast + stagger:
            time.sleep(tlast + stagger - now)
        tlast = time.time()
        
        env = {}
        env.update(os.environ)
        env.update({
                "SPAWNER_PROCESS_NUMBER":"%s" % (nstarted,),
                "SPAWNER_TOTAL_PROCESSES":"%s" % (NRUN,)
            })

        args = command[:]
        for i, a in enumerate(args):
            if a == "__i":  args[i] = str(nstarted)
        
        p = subprocess.Popen(args, stdout=output, stderr=output,
            env=env)
        Pipes.append(p)
        nstarted += 1
        print "done/started: %d/%d" % (ndone, nstarted)
    for p in Pipes:
        if p.poll() != None:
            Pipes.remove(p)
            ndone += 1
            print "done/started: %d/%d" % (ndone, nstarted)

while Pipes:
    for p in Pipes:
        if p.poll() != None:
            Pipes.remove(p)
            ndone += 1
            print "done/started: %d/%d" % (ndone, nstarted)

t = time.time() - t0   
print "time=", t   
print "rate=", NRUN/t 
        
    



