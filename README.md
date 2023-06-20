# Nematode-Connectome-Weight-Inference
My graduation thesis (cooperated with Dr. Mengdi Zhao). 

Aim to use behavior supervised data to train a functional connectome weight for detailed compartmental model of nematode C.elegans

# Pipeline

1. Build detailed multi-compartmental model for Nematode C.elegans (see ```network/```)
2. Train ANN for every neuron of C.elegans, fitting I/O mapping (see ```single_nrn_train/```)
3. Train connectome weight parameter with respect to specific behavior target (see ```network_train/```)