import cPickle as pickle
import test_script
import execnet
import os

gw = execnet.makegateway() #"ssh=10.155.208.100")
ch = gw.remote_exec(test_script)
for root, dirs, fileids in os.walk("/media/tiny"):
    for fileid in fileids:
		print fileid
		ch1 = gw.newchannel()
		tmp = {'a': 1, 'b': 3}
		ch.send(pickle.dumps(fileid))
		ch.send(ch1)
		ch1.send(pickle.dumps(tmp))
		print pickle.loads(ch1.receive())