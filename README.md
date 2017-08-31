# HDPGM C code

C code for HDPGM parallel computing. Currently, the parallel version is only available on Linux and MacOS.

## About HDPGM
HDPGM is short for *the hiearachical and dynamic Poisson graphical model*. The model was proposed by Xiangyu Luo and Yingying Wei to recover transcription factor (TF) networks from ChIP-Seq count data. On the one hand, TF networks are heterogeneous across the genome. On the other hand, TF networks are dynamic with respect to different cell types. Under the mild assumption that the whole genome can be partitioned into several (unknown) sets and genomic locations in the same set share the same TF networks, HDPGM can be used to simultaneously construct the dynamic TF networks and cluster genomic locations without need to pre-specify the number of  clusters.

The Markov chain Monte Carlo algorithm (MCMC) was used to draw posterior samples from HDPGM. For more technical details, please refer to our paper **Nonparametric Bayesian Learning of Heterogeneous Dynamic Transcription Factor Networks** (under revision).   

## Parallel Computing
The idea of the message passing interface (MPI) was employed to parallelize the MCMC algorithm. Among MPI functions, MPI\_Rec and MPI\_Send were used the most frequently in the C code. 

## Implementation
Assume there are G genomic locations, D conditions, p TFs, R replicates.
1. Download the source file **HDPGM.c** and the header file **HDPGM.h** in your working directory. 
2. Prepare your data file (a G by D\*p\*R matrix) where rows correspond to G genomic locations. For the columns, column 1 corresponds to condition 1, TF 1, replicate 1, which are coded as (1,1,1). Column 2 is (1,1,2), ... , column R is (1,1,R), column R+1 is (1,2,1), column p\*R is (1,p,R), column p\*R+1 is (2,1,1) and so on so forth until column D\*p\*R which is (D, p, R).
3. Name your data file as "Data.txt" and put it in your working directory.
4. In your working directory, make a new directory "PosteriorSamples". Under the "PosteriorSamples", make four new directories: "P\_dp\_t", "cla\_t", "L\_t" and "Lambda\_t".
5. In the source file **HDPGM.c**, set G, D, p, R, and the maximum cluster number **M**.
4. Open the terminal, change the current diretory to your working directory and input the following two commands. Note that 10 threads are used here, and you could flexibly set it.

```
mpicc HDPGM.c -o HDPGM -lm -std=c11
mpirun -np 10 ./HDPGM
```

## Remarks
1. The number of threads must be greater than one. 
2. If G cannot be divisible by the thread number L, the first L-1 threads have \[G/L\] genomic locations each and the last thread has G - (L-1)\[G/L\] genomic locations, where \[x\] is the largest integer less than or equal to x.  
3. When the program is running, there will be a "log.txt" file showing the running progress.  
