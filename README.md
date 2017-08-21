# HDPGM C code

C code for HDPGM parallel computing

## About HDPGM
HDPGM is short for *the hiearachical and dynamic Poisson graphical model*. The model was proposed by Xiangyu Luo and Yingying Wei to recover transcription factor (TF) networks from ChIP-Seq count data. On the one hand, TF networks are heterogeneous across the genome. On the other hand, TF networks are dynamic with respect to different cell types. Under the mild assumption that the whole genome can be partitioned into several (unknown) sets and genomic locations in the same set share the same TF networks, HDPGM can be used to simultaneously construct the dynamic TF networks and cluster genomic locations without need to pre-specify the number of  clusters. For more technical details, please refer to our paper *Nonparametric Bayesian Learning of Heterogeneous Dynamic Transcription Factor Networks* (under revision).    
