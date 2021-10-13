import cProfile, pstats, sys
import magic.aklt_test

reps = 20
pr = cProfile.Profile()
pr.enable()
for i in range(reps):
    magic.aklt_test.main()

pr.disable()
ps = pstats.Stats(pr, stream=sys.stdout)
t1 = ps.total_tt

pr.enable()
magic.aklt_test.main(steps=reps)
pr.disable()
ps = pstats.Stats(pr, stream=sys.stdout)
t2 = ps.total_tt - t1
print([t1, t2])
