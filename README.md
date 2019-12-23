# Active-Memory
Repository for the paper: 'Is Attention All What You Need ? - An Empirical Investigation on  Convolution-Based Active Memory and Self-Attention'

Abstract:
The key to a Transformer model is the self-attention mechanism, which allows the model to analyze an entire sequence in a computationally efficient manner. Recent work has suggested the possibility that general attention mechanisms used by RNNs could be replaced by active-memory mechanisms. In this work, we evaluate whether various active-memory mechanisms could replace self-attention in a Transformer. Our experiments suggest that active-memory alone achieves comparable results to the self-attention mechanism for language modelling, but optimal results are mostly achieved by using both active-memory and self-attention mechanisms together. We also note that, for some specific algorithmic tasks, active-memory mechanisms alone outperform both the active memory and a combination of the two. 

##To run WT3 experiments, run WT3/main.py. Download WT3 dataset beforehand and place in WT3/data/

##To run algorithmic tasks, run Algorithmic-Tasks/main.py
