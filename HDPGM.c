#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include "HDPGM.h"
#include <mpi.h>


//Note: in this version, Y_t has the dimenstion: N, D, R, p, p and X_t has the dimension: N, D, R, p
int main(){
    MPI_Init(NULL, NULL);
    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);//world_size = #cores
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);//world_rank = 0,1,...,#cores-1
    int lgth; //block size
    int lgth_last; //the size for the last block
    
    
    srand(25082017); //set seed
	
	//=============================================================================================================
	//The parameters you have to input
    int T = 100000; // number of iterations
    int S = 50000;  //the iteration number at which we begin to collect posterior samples
    int thin = 5;   // the thin; in this example, we collect sample 50001, 50006, 50011, ..., 99996
    int D=3;	    // the number of conditions
	int p=30;		// the number of TFs
	int N = 22402;  // the number of genomic locations
	int M = 20;     // the maximum cluster number
	int R=2;        // the number of replicates
    //==============================================================================================================
    lgth = N / world_size;
    lgth_last = N - lgth * (world_size-1);

    if((world_size == 1)&(world_rank == 0)){
	fprintf(stderr, "\n=========================================================================================\n");
	fprintf(stderr, "=   Number of threads must be greater than one!\n");
	fprintf(stderr, "=========================================================================================\n\n");
      	exit(-1);
    }else if(world_rank==0){
	printf("=========================================================================================\n");
	printf("=   There are %d threads.\n", world_size);
	printf("=   For thread 0 to %d, each of them processes data on %d genomic locations.\n", world_size-2, lgth);
	printf("=   For thread %d, it processes data on %d genomic locations.\n", world_size-1, lgth_last);
	printf("=========================================================================================\n\n");
    }

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
        
        fp=fopen("Data.txt","r");
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
    	printf("Input done!\n");
  


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
                for(int r=0;r<R;r++){
                	for(int i=0; i<p; i++){
                		for(int j=i; j<p; j++){
                			if(j==i){
                				Y_t[n][d][r][i][j] = X[n][d][r][i];
							}else{
								Y_t[n][d][r][i][j] = 0;
								Y_t[n][d][r][j][i] = 0;
							}
						}
					}
                    
                }
                
            }
        }
    }//the end of if(world_rank == 0)

    int *****Y_t_block; // each block corresponds to one core
    if(world_rank < world_size-1){
    	Y_t_block = make5Darray_int_continuousMem(lgth, D, R, p, p);
    }else{
		Y_t_block = make5Darray_int_continuousMem(lgth_last, D, R, p, p);
    }


    int *cla_t_block;
    if(world_rank < world_size-1){
    	cla_t_block = (int *) malloc(lgth * sizeof(int));
    }else{
    	cla_t_block = (int *) malloc(lgth_last * sizeof(int));
    }

    if(world_rank == 0){
        //send blocks of cla_t and Y_t to thread 1 to #cores-1
        for(int nthread = 1; nthread < world_size; nthread++){
	    	if(nthread < world_size-1){
            	MPI_Send(&(cla_t[lgth*nthread]),lgth,MPI_INT,nthread,222,MPI_COMM_WORLD);
            	MPI_Send(&(Y_t[lgth*nthread][0][0][0][0]),lgth*D*R*p*p,MPI_INT,nthread,333,MPI_COMM_WORLD);
	    	}else{
            	MPI_Send(&(cla_t[lgth*nthread]),lgth_last, MPI_INT,nthread,222,MPI_COMM_WORLD);
            	MPI_Send(&(Y_t[lgth*nthread][0][0][0][0]),lgth_last*D*R*p*p,MPI_INT,nthread,333,MPI_COMM_WORLD);
	    	}
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
		if(world_rank < world_size-1){
        	MPI_Recv(&(cla_t_block[0]),lgth,MPI_INT,0,222,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        	MPI_Recv(&(Y_t_block[0][0][0][0][0]), lgth*D*R*p*p, MPI_INT, 0, 333, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	}else{
        	MPI_Recv(&(cla_t_block[0]),lgth_last,MPI_INT,0,222,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        	MPI_Recv(&(Y_t_block[0][0][0][0][0]), lgth_last*D*R*p*p, MPI_INT, 0, 333, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if(world_rank==0){
	printf("Conducting Gibbs Sampler...\n");
    }
    
    //======================================================================================================
    //Gibbs sampler
    //======================================================================================================

    int len, mark;
    char filename[100];
    char ind[10];
    double t1, t2;

    t1 = MPI_Wtime();	
    for (int t=1;t<=T;t++){
        if(world_rank == 0){
	    
            /* sample network structures and intensity parameters Lambda_t & L_t*/
            if(t==1){
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

		if(world_rank < world_size - 1){
        	for (int n=0;n<lgth;n++){
            		cla_t_block[n]=sample_cla(M, D, p, R, n, Lambda_t,P_dp_t,Y_t_block);
        	}
		}else{
        	for (int n=0;n<lgth_last;n++){
            		cla_t_block[n]=sample_cla(M, D, p, R, n, Lambda_t,P_dp_t,Y_t_block);
        	}		
		}
        
        MPI_Barrier(MPI_COMM_WORLD);
        /* sample latent counts Y_t_block*/
		if(world_rank < world_size - 1){
        	for(int n=0; n<lgth; n++){
            		for(int d=0;d<D;d++){
                		for (int r=0;r<R;r++){
                    			sample_Y(d,r,n,p,Lambda_t,cla_t_block,Y_t_block);
                		}
            		}
        	}
		}else{
        	for(int n=0; n<lgth_last; n++){
            		for(int d=0;d<D;d++){
                		for (int r=0;r<R;r++){
                    			sample_Y(d,r,n,p,Lambda_t,cla_t_block,Y_t_block);
                		}
            		}
        	}
		}
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        //assemble updates in blocks
        if((world_rank > 0)&(world_rank < world_size-1)){
            MPI_Send(cla_t_block, lgth, MPI_INT, 0, 444, MPI_COMM_WORLD);
            MPI_Send(&(Y_t_block[0][0][0][0][0]), lgth*D*R*p*p, MPI_INT, 0, 555,MPI_COMM_WORLD);
        }

		if((world_rank > 0)&(world_rank == world_size-1)){
            MPI_Send(cla_t_block, lgth_last, MPI_INT, 0, 444, MPI_COMM_WORLD);
            MPI_Send(&(Y_t_block[0][0][0][0][0]), lgth_last*D*R*p*p, MPI_INT, 0, 555,MPI_COMM_WORLD);
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
				if(nthread < world_size-1){
                	MPI_Recv(&(cla_t[nthread*lgth]), lgth, MPI_INT, nthread, 444, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                	MPI_Recv(&(Y_t[nthread*lgth][0][0][0][0]), D*R*lgth*p*p, MPI_INT, nthread, 555, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            	}else{
                	MPI_Recv(&(cla_t[nthread*lgth]), lgth_last, MPI_INT, nthread, 444, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                	MPI_Recv(&(Y_t[nthread*lgth][0][0][0][0]), D*R*lgth_last*p*p, MPI_INT, nthread, 555, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
	    	}
        
        
            //Gibbs sampler visulization
            /*printf("Iteration: %d\n", t);
        
            for(int i=0 ; i<M; i++){
                printf("cluster %d:\t%f\n", i, P_dp_t[i]);
            }
            printf("\n");*/

            len=unique(N, cla_t,cla_star);

            /*for(int m=0;m<M;m++){
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
            }*/

            //////////////////////
            //write files
            /////////////////////
	    	if((t>=S)&(t%thin==1)){//begin if
            	strcpy(filename, "PosteriorSamples/Lambda_t/Lambda_t_");
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
        
            	strcpy(filename, "PosteriorSamples/L_t/L_t_");
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
        
            	strcpy(filename, "PosteriorSamples/P_dp_t/P_dp_t_");
            	sprintf(ind, "%d", t);
            	strcat(filename, ind);
            	strcat(filename, ".txt");
            	fp = fopen(filename, "w");
            	for(int m=0; m<M; m++){
                	fprintf(fp, "%f\t", P_dp_t[m]);
            	}
	    		fprintf(fp, "\n");
            	fclose(fp);
        
        
            	strcpy(filename, "PosteriorSamples/cla_t/cla_t_");
            	sprintf(ind, "%d", t);
            	strcat(filename, ind);
            	strcat(filename, ".txt");
            	fp = fopen(filename, "w");
            	for(int n=0; n<N; n++){
                	fprintf(fp, "%d\t", cla_t[n]);
            	}
	    		fprintf(fp, "\n");
            	fclose(fp);  
	    	}//end if      
        }

	if(world_rank==0){
	    //record the how many clusters are there for each iteration
        fp = fopen("K_t.txt", "a");
	    fprintf(fp, "Iteration: %d\n", t);
        fprintf(fp, "cluster number: %d\n", len);
	    for(int m=0; m<M; m++){
			fprintf(fp, "%f\t", P_dp_t[m]);
	    }
        fprintf(fp, "\n");
	    /*mark = 0;
        for(int i=0; i < len; i++){
            if(cla_star[i] == 2){ //2 corresponds to table 3.
                mark = 1;
                break;
            }
    	}
        if(mark == 1){                
            for(int i=0;i<p;i++){
                for(int j=0;j<p;j++){
                    fprintf(fp, "%d ",L_t[2][0][i][j]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        } */   
        fclose(fp);
	}

        MPI_Barrier(MPI_COMM_WORLD);
    }

    t2 = MPI_Wtime();
    if(world_rank==0){
		printf("Output done!\n");
		printf("Cost %f seconds.", t2 - t1);
    }
    free(P_dp_t);
    delet4Darray_continuousMem(Lambda_t, M , D, p, p);
    free(cla_t_block);
    if(world_rank < world_size -1){
    	delet5Darray_int_continuousMem(Y_t_block, lgth, D, R, p, p);
    }else{
    	delet5Darray_int_continuousMem(Y_t_block, lgth_last, D, R, p, p);
    }

    if(world_rank == 0){
        free(cla_t);
        free(cla_star);
        delet5Darray_int_continuousMem(Y_t, N, D, R, p, p);
        delet4Darray_int_continuousMem(X, N, D, R, p);
        delet4Darray_int_continuousMem(L_t, M, D, p, p);
    }
    
    MPI_Finalize();
    return 0;
}
