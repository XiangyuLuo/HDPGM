# HDPGM C code

C code for HDPGM parallel computing

## About HDPGM
HDPGM is short for *the hiearachical and dynamic Poisson graphical model*. The model was proposed by Xiangyu Luo and Yingying Wei to recover transcription factor (TF) networks from ChIP-Seq count data. On the one hand, TF networks are heterogeneous across the genome. On the other hand, TF networks are dynamic with respect to different cell types. Under the mild assumption that the whole genome can be partitioned into several (unknown) sets and genomic locations in the same set share the same TF networks, HDPGM can be used to simultaneously construct the dynamic TF networks and cluster genomic locations without need to pre-specify the number of  clusters.

The Markov chain Monte Carlo algorithm (MCMC) was used to draw posterior samples from HDPGM. For more technical details, please refer to our paper *Nonparametric Bayesian Learning of Heterogeneous Dynamic Transcription Factor Networks* (under revision).   

## Parallel Computing
The idea of the message passing interface (MPI) was employed to parallelize the MCMC algorithm. MPI\_Rec and MPI\_Send were used the most frequently in the C code. 

## Implementation
Assume there are G genomic locations, D conditions, p TFs, R replicates.
1. Download the source file **HDPGM.c** and the header file **HDPGM.h** in your work folder. 
2. Prepare your data file (a G by D\*p\*R matrix) where rows correspond to genomic locations. For the columns, column 1 corresponds to the observed count data   
