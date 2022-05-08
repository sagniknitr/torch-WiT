from pruning.experiment import PruningExperiment

import os
os.environ['DATAPATH'] = '/path/to/data'

for strategy in ['RandomPruning', 'GlobalMagWeight', 'LayerMagWeight']:
    for  c in [1,2,4,8,16,32,64]:
        exp = PruningExperiment(dataset='MNIST', 
                                model='MnistNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':10})
        exp.run()