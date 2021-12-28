import numpy as np
import pickle
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/')
import magicRenyi as renyi
# import gc


def main(steps=1):
    d = 3
    n = 8
    stepSize = 0.1
    jRange = 2 / 3
    Js = [0.1] #[np.round(stepSize * i, 3) for i in range(int(jRange / stepSize))]
    for i in range(len(Js)):
       J = Js[i]
       with open('magic/results/haldane/psi_haldane_J_' + str(np.round(J, 5)) + '_16', 'rb') as f:
           gs = pickle.load(f)
       renyi.getSecondRenyiFromRandomVecs(gs, d, outdir='results/haldane_J_' + str(np.round(J, 5)),
                                          rep=2, speedup=True, steps=steps)
