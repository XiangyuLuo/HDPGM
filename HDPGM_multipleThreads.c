#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include "HDPGM_multipleThreads.h"
#include <mpi.h>


//Note: in this version, Y_t has the dimenstion: N, D, R, p, p and X_t has the dimension: N, D, R, p
int main(){
    MPI_Init(NULL, NULL);
    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);//world_size = #cores
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);//world_rank = 0,1,...,#cores-1
    int lgth; //block size
    
    
    srand(8112017); //set seed
    int T = 40000;
    int K=5, D=3, p=20, N = 10000, M = 10, R=2;
    lgth = N / world_size;

    int *cla_t, *cla_star, *****Y_t, ****X;
    
    if(world_rank==0){
        cla_t = (int *)malloc(N*sizeof(int));
        cla_star = (int *)malloc(N*sizeof(int));
        for(int n=0; n < N; n++){
            cla_star[n] = -1;
        }
        Y_t = make5Darray_int_continuousMem(N, D, R, p, p);
        X = make4Darray_int_continuousMem(N, D, R, p);
    }

    double alpha = 2;
    double *alpha_prob, *P_dp_t;
    P_dp_t = (double *)malloc(M*sizeof(double));
    if(world_rank == 0){
        alpha_prob = (double *)malloc(M*sizeof(double));
    }
    
    double ****Lambda_t;
    int ****L_t;
    Lambda_t = make4Darray_continuousMem(M , D, p, p);
    if(world_rank==0){
        
        L_t = make4Darray_int_continuousMem(M, D, p, p);
    }

    double par_p_t=1.0/4;
    double par_0[2], par_1[2], par_2[2];
    par_0[0] = 2;
    par_0[1] = 20;
    par_1[0] = 2;
    par_1[1] = 1;
    par_2[0] = 3;
    par_2[1] = 1;
    FILE* fp;
    double *prob;

    if(world_rank==0){
    
        
        fp=fopen("X_matr.txt","r");
        for (int n=0;n<N;n++){
            for (int d=0;d<D;d++){
                for (int i=0;i<p;i++){
                    for (int r=0;r<R;r++){
				        fscanf(fp,"%d", &(X[n][d][r][i]));
                    }
                }
            }
        }
        fclose(fp);
    	printf("input done\n");
  


        for (int m=0;m<M;m++){
            alpha_prob[m]=2.0 / M;
        }
    
        //set initial values
        rDirich(alpha_prob,P_dp_t, M);
    
        for (int m=0;m<M;m++){
            sample_H(D, p, par_0,par_1,par_2,par_p_t,L_t[m],Lambda_t[m]);
        }
    
    
        prob = (double *)malloc(M*sizeof(double));
        for(int m=0;m<M;m++){
            prob[m]=1.0/M;
        }
    
        for(int n=0;n<N;n++){
            cla_t[n]=rdiscrete(M, prob);
        }
        free(prob);
    
    
    
        for(int d=0;d<D;d++){
            for(int n=0;n<N;n++){
                for(int i=0;i<p;i++){
                    for(int r=0;r<R;r++){
                        for(int i=0; i<p; i++){
                            for(int j=0; j<p; j++){
                                if(j==i){
                                    Y_t[n][d][r][i][j]=X[n][d][r][i];
                                }else{
                                    Y_t[n][d][r][i][j]=0;
                                }
                            }
                        }
                    
                    }
                }
            }
        }
    }//the end of if(world_rank == 0)

    int *****Y_t_block; // each block corresponds to one core
    Y_t_block = make5Darray_int_continuousMem(lgth, D, R, p, p);
    int *cla_t_block;
    cla_t_block = (int *) malloc(lgth * sizeof(int));
    

    if(world_rank == 0){
        //send blocks of cla_t and Y_t to thread 1 to #cores-1
        for(int nthread = 1; nthread < world_size; nthread++){
            MPI_Send(&(cla_t[lgth*nthread]),lgth,MPI_INT,nthread,222,MPI_COMM_WORLD);
            MPI_Send(&(Y_t[lgth*nthread][0][0][0][0]),lgth*D*R*p*p,MPI_INT,nthread,333,MPI_COMM_WORLD);
        }
        for(int n=0; n < lgth; n++){
            cla_t_block[n] = cla_t[n];
            for (int d=0;d<D;d++){
                for (int i=0;i<p;i++){
                    for (int j=0;j<p;j++){
                        for (int r=0;r<R;r++){
                            Y_t_block[n][d][r][i][j] = Y_t[n][d][r][i][j];
                        }
                    }
                }
            }
        }
    }else{
        //receive messages from thread 0
        MPI_Recv(&(cla_t_block[0]),lgth,MPI_INT,0,222,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&(Y_t_block[0][0][0][0][0]), lgth*D*R*p*p, MPI_INT, 0, 333, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    
    
    //======================================================================================================
    //Gibbs sampler
    //======================================================================================================
    printf("begin gibbs sampler:\n");
    int len, mark;
    char filename[100];
    char ind[10];
    double t1, t2;

    t1 = MPI_Wtime();	
    for (int t=0;t<T;t++){
        if(world_rank == 0){
	    
            /* sample network structures and intensity parameters Lambda_t & L_t*/
            if(t==0){
                len=unique(N, cla_t,cla_star);
            }
            for(int m=0;m<M;m++){
                mark=0;
                for(int n=0;n<len;n++){
                    if (cla_star[n]==m){
                        mark=1;
                        break;
                    }
                }
                if(mark==0) {                   //mark==0 means m is in cla_diff//
                    sample_H(D, p, par_0, par_1, par_2,par_p_t,L_t[m], Lambda_t[m]);
                }else {            //mark==1 means m is in cla_star//
                    sample_L_m(D, p, Lambda_t[m],par_p_t,par_0,par_1, L_t[m]);
                    sample_Lambda_m(D, R, N, p, Y_t,L_t,cla_t,m,par_0,par_1,par_2,Lambda_t[m]);
                }
            }
            /* sample cluster proportion */
            sample_P_dp(N, M, cla_t,alpha,P_dp_t);

        }

        //broadcast P_dp_t and Lambda_t to all other threads
	if(world_rank == 0){
		for(int nthread=1; nthread < world_size; nthread++){
        		MPI_Send(&(Lambda_t[0][0][0][0]), M*D*p*p, MPI_DOUBLE, nthread, 222, MPI_COMM_WORLD);
        		MPI_Send(P_dp_t, M, MPI_DOUBLE, nthread, 333, MPI_COMM_WORLD);
		}
        }else{
		MPI_Recv(&(Lambda_t[0][0][0][0]), M*D*p*p, MPI_DOUBLE, 0, 222, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(P_dp_t, M, MPI_DOUBLE, 0, 333, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

        /* sample class membership cla_t_block*/
        MPI_Barrier(MPI_COMM_WORLD);

        for (int n=0;n<lgth;n++){
            cla_t_block[n]=sample_cla(M, D, p, R, n, Lambda_t,P_dp_t,Y_t_block);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        /* sample latent counts Y_t_block*/
        for(int n=0; n<lgth; n++){
            for(int d=0;d<D;d++){
                for (int r=0;r<R;r++){
                    sample_Y(d,r,n,p,Lambda_t,cla_t_block,Y_t_block);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        //assemble updates in blocks
        if(world_rank != 0){
            MPI_Send(cla_t_block, lgth, MPI_INT, 0, 444, MPI_COMM_WORLD);
            MPI_Send(&(Y_t_block[0][0][0][0][0]), lgth*D*R*p*p, MPI_INT, 0, 555,MPI_COMM_WORLD);
        }
        
        if(world_rank == 0){
            for(int n=0; n<lgth; n++){
                cla_t[n] = cla_t_block[n];
                for(int d=0;d<D;d++){
                    for (int r=0;r<R;r++){
                        for(int i=0; i<p; i++){
                            for(int j=0; j<p; j++){
                                Y_t[n][d][r][i][j] = Y_t_block[n][d][r][i][j];
                            }
                        }
                    }
                }
            }
            
            for(int nthread = 1; nthread < world_size; nthread++){
                MPI_Recv(&(cla_t[nthread*lgth]), lgth, MPI_INT, nthread, 444, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&(Y_t[nthread*lgth][0][0][0][0]), D*R*lgth*p*p, MPI_INT, nthread, 555, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        
        
            //Gibbs sampler visulization
            printf("Iteration: %d\n", t);
        
            for(int i=0 ; i<M; i++){
                printf("cluster %d:\t%f\n", i, P_dp_t[i]);
            }
            printf("\n");
            len=unique(N, cla_t,cla_star);
            for(int m=0;m<M;m++){
                mark = 0;
                for(int i=0; i < len; i++){
                    if(cla_star[i] == m){
                        mark = 1;
                        break;
                    }
                }
                if(mark == 1){
                    printf("cluster %d:\n", m);
                    for(int d=0;d<D;d++){
                        for(int i=0;i<p;i++){
                            for(int j=0;j<p;j++){
                                printf("%d ",L_t[m][d][i][j]);
                            }
                            printf("\n");
                        }
                        printf("\n");
                    }
                }
                printf("\n");
            }
            //////////////////////
            //write files
            /////////////////////
            strcpy(filename, "HDPGM_multipleThreads_simulation/Lambda_t/Lambda_t_");
            sprintf(ind, "%d", t);
            strcat(filename, ind);
            strcat(filename, ".txt");
            fp = fopen(filename, "w");
            for(int m=0; m<M; m++){
                for (int d=0;d<D;d++){
                    for (int i=0;i<p;i++){
                        for (int j=0;j<p;j++){
                            fprintf(fp, "%f\t", Lambda_t[m][d][i][j]);
                        }
                        fprintf(fp, "\n");
                    }
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        
            strcpy(filename, "HDPGM_multipleThreads_simulation/L_t/L_t_");
            sprintf(ind, "%d", t);
            strcat(filename, ind);
            strcat(filename, ".txt");
            fp = fopen(filename, "w");
            for(int m=0; m<M; m++){
                for (int d=0;d<D;d++){
                    for (int i=0;i<p;i++){
                        for (int j=0;j<p;j++){
                            fprintf(fp, "%d\t", L_t[m][d][i][j]);
                        }
                        fprintf(fp, "\n");
                    }
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        
            strcpy(filename, "HDPGM_multipleThreads_simulation/P_dp_t/P_dp_t_");
            sprintf(ind, "%d", t);
            strcat(filename, ind);
            strcat(filename, ".txt");
            fp = fopen(filename, "w");
            for(int m=0; m<M; m++){
                fprintf(fp, "%f\t", P_dp_t[m]);
            }
            fclose(fp);
        
        
            strcpy(filename, "HDPGM_multipleThreads_simulation/cla_t/cla_t_");
            sprintf(ind, "%d", t);
            strcat(filename, ind);
            strcat(filename, ".txt");
            fp = fopen(filename, "w");
            for(int n=0; n<N; n++){
                fprintf(fp, "%d\t", cla_t[n]);
            }
            fclose(fp);
        
            //record the how many clusters are there for each iteration
            fp = fopen("HDPGM_oneThread_K_t.txt", "a");
            fprintf(fp, "%d\n", len);
            fclose(fp);
        
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    t2 = MPI_Wtime();
    free(P_dp_t);
    delet4Darray_continuousMem(Lambda_t, M , D, p, p);
    free(cla_t_block);
    delet5Darray_int_continuousMem(Y_t_block, lgth, D, R, p, p);
    
    if(world_rank == 0){
	printf("\nThe time difference is %f seconds.\n", t2 - t1);
        free(cla_t);
        free(cla_star);
        delet5Darray_int_continuousMem(Y_t, N, D, R, p, p);
        delet4Darray_int_continuousMem(X, N, D, R, p);
    
        delet4Darray_int_continuousMem(L_t, M, D, p, p);
    }
    
    MPI_Finalize();
    return 0;
}
