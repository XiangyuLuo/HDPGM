/*Basic functions*/

//construct multi-dimensional arrays with contiguous memories 

double** make2Darray_continuousMem(int a1, int a2){
	double *tmp_vec = (double *)malloc(a1*a2*sizeof(double));
	double **tmp;
	tmp = (double **)malloc(a1*sizeof(double *));
	for(int i=0; i<a1; i++){
		tmp[i] = &(tmp_vec[i*a2]);
	}
	return tmp;
}

void delet2Darray_continuousMem(double **tmp, int a1, int a2){
	free(tmp[0]);
	free(tmp);
}

int** make2Darray_int_continuousMem(int a1, int a2){
    int *tmp_vec = (int *)malloc(a1*a2*sizeof(int));
    int **tmp;
    tmp = (int **)malloc(a1*sizeof(int *));
    for(int i=0; i<a1; i++){
        tmp[i] = &(tmp_vec[i*a2]);
    }
    return tmp;
}

void delet2Darray_int_continuousMem(int **tmp, int a1, int a2){
    free(tmp[0]);
    free(tmp);
}

double*** make3Darray_continuousMem(int a1, int a2, int a3){
	double *tmp_vec = (double *)malloc(a1*a2*a3*sizeof(double));
	double ***tmp;
	tmp = (double ***)malloc(a1*sizeof(double **));
	for(int i=0; i<a1; i++){
		tmp[i] = (double **) malloc(a2*sizeof(double *));
		for(int j=0; j<a2; j++){
			tmp[i][j] = &(tmp_vec[i*a2*a3+j*a3]);
		}
	}
	return tmp;
}

void delet3Darray_continuousMem(double ***tmp, int a1, int a2, int a3){
	free(tmp[0][0]);
	for(int i=0; i<a1; i++){
		free(tmp[i]);
	}
	free(tmp);
}

double**** make4Darray_continuousMem(int a1, int a2, int a3, int a4){
	double *tmp_vec = (double *)malloc(a1*a2*a3*a4*sizeof(double));
	double ****tmp;
	tmp = (double ****)malloc(a1*sizeof(double ***));
	for(int i=0; i<a1; i++){
		tmp[i] = (double ***) malloc(a2*sizeof(double **));
		for(int j=0; j<a2; j++){
			tmp[i][j] = (double **) malloc(a3*sizeof(double *));
			for(int k=0; k<a3; k++){
				tmp[i][j][k] = &(tmp_vec[i*a2*a3*a4+j*a3*a4+k*a4]);
			}
		}
	}
	return tmp;
}

void delet4Darray_continuousMem(double ****tmp, int a1, int a2, int a3, int a4){
	free(tmp[0][0][0]);
	for(int i=0; i<a1; i++){
		for(int j=0; j<a2; j++){
			free(tmp[i][j]);			
		}
		free(tmp[i]);
	}
	free(tmp);
}

int**** make4Darray_int_continuousMem(int a1, int a2, int a3, int a4){
    int *tmp_vec = (int *)malloc(a1*a2*a3*a4*sizeof(int));
    int ****tmp;
    tmp = (int ****)malloc(a1*sizeof(int ***));
    for(int i=0; i<a1; i++){
        tmp[i] = (int ***) malloc(a2*sizeof(int **));
        for(int j=0; j<a2; j++){
            tmp[i][j] = (int **) malloc(a3*sizeof(int *));
            for(int k=0; k<a3; k++){
                tmp[i][j][k] = &(tmp_vec[i*a2*a3*a4+j*a3*a4+k*a4]);
            }
        }
    }
    return tmp;
}

void delet4Darray_int_continuousMem(int ****tmp, int a1, int a2, int a3, int a4){
    free(tmp[0][0][0]);
    for(int i=0; i<a1; i++){
        for(int j=0; j<a2; j++){
            free(tmp[i][j]);			
        }
        free(tmp[i]);
    }
    free(tmp);
}


double***** make5Darray_continuousMem(int a1, int a2, int a3, int a4, int a5){
	double *tmp_vec = (double *)malloc(a1*a2*a3*a4*a5*sizeof(double));
	double *****tmp;
	tmp = (double *****)malloc(a1*sizeof(double ****));
	for(int i=0; i<a1; i++){
		tmp[i] = (double ****) malloc(a2*sizeof(double ***));
		for(int j=0; j<a2; j++){
			tmp[i][j] = (double ***) malloc(a3*sizeof(double **));
			for(int k=0; k<a3; k++){
				tmp[i][j][k] = (double **) malloc(a4*sizeof(double *));
				for(int ell=0; ell<a4; ell++){
					tmp[i][j][k][ell] = &(tmp_vec[i*a2*a3*a4*a5+j*a3*a4*a5+k*a4*a5+ell*a5]);					
				}
			}
		}
	}
	return tmp;
}

void delet5Darray_continuousMem(double *****tmp, int a1, int a2, int a3, int a4, int a5){
	free(tmp[0][0][0][0]);
	for(int i=0; i<a1; i++){
		for(int j=0; j<a2; j++){
			for(int k=0; k<a3; k++){
				free(tmp[i][j][k]);
			}
			free(tmp[i][j]);			
		}
		free(tmp[i]);
	}
	free(tmp);
}

int***** make5Darray_int_continuousMem(int a1, int a2, int a3, int a4, int a5){
    int *tmp_vec = (int *)malloc(a1*a2*a3*a4*a5*sizeof(int));
    int *****tmp;
    tmp = (int *****)malloc(a1*sizeof(int ****));
    for(int i=0; i<a1; i++){
        tmp[i] = (int ****) malloc(a2*sizeof(int ***));
        for(int j=0; j<a2; j++){
            tmp[i][j] = (int ***) malloc(a3*sizeof(int **));
            for(int k=0; k<a3; k++){
                tmp[i][j][k] = (int **) malloc(a4*sizeof(int *));
                for(int ell=0; ell<a4; ell++){
                    tmp[i][j][k][ell] = &(tmp_vec[i*a2*a3*a4*a5+j*a3*a4*a5+k*a4*a5+ell*a5]);
                }
            }
        }
    }
    return tmp;
}

void delet5Darray_int_continuousMem(int *****tmp, int a1, int a2, int a3, int a4, int a5){
    free(tmp[0][0][0][0]);
    for(int i=0; i<a1; i++){
        for(int j=0; j<a2; j++){
            for(int k=0; k<a3; k++){
                free(tmp[i][j][k]);
            }
            free(tmp[i][j]);			
        }
        free(tmp[i]);
    }
    free(tmp);
}
/*generate random numbers */

double runif(){ //sample a random number from 0 to 1
	double temp;
	do{
		temp = (double) rand() / (double) RAND_MAX;
	}while(temp>=1 || temp <=0);
	return temp;
} 

int rpoiss(double lambda) { //proposed by Knuth
    double L = exp(-lambda);
    int k = 0; 
    double tmp=1.0;  
    double u;
    do {
        k+=1;
        u = runif();
        tmp *=u;
    } while (tmp>L);
    return(k-1);
}

double logfactorial(int x){ //calculate log(x!)
    double tmp=0.0;
    if(x <= 0){
        return 0.0;
    }else{
        for (int i=1;i<=x;i++){
            tmp += log(i);
        }
        return tmp;
    }
}

double log_dpoiss(int x, double lambda){ //calculate the log of poisson density at x with mean lambda
    return x*log(lambda)-lambda-logfactorial(x);
}

int rbernoulli(double prob) { // prob must be between 0 and 1
    double u = runif();
    if(u < prob){
    	return 1;
	}else{
	    return 0;		
	}
}

double rexp(double lambda){
    double u = runif();
    return (-log(u)/lambda);
}

double rgamma(double shape, double rate){
	double scale = 1.0 / rate;
	int shape_int = floor(shape);
	double s = 0;
	for(int i = 0; i < shape_int; i++){
		s = s - log(runif());  
	}
	
	if(shape_int == shape){
		return scale * s;	
	}else{
		double U, V, W, xi, eta;
		double delta = shape - shape_int; 
		do{
			U = runif();
			V = runif();
			W = runif();
			if(U <= exp(1) / (exp(1) + delta)){
				xi = pow(V, 1 / delta);
				eta = W * pow(xi, delta - 1);
			}else{
				xi = 1 - log(V);
				eta = W * exp(-xi);
			}
		}while(eta > pow(xi, delta-1)*exp(-xi));
		return scale * ( xi + s); 
	}
}

double log_dgamma(double x,double shape, double rate){
    return shape*log(rate)+(shape-1)*log(x)-rate*x-lgamma(shape);
}

double rbeta(double alpha, double beta){
    double x=rgamma(alpha, 1.0);
    double y=rgamma(beta, 1.0);
    return x/(x+y);
}

int rdiscrete(int M, double *prob_original){ //sample from 0 to M-1 with prob. prob_original
    double sum=0;
    int ans;
    double *prob = (double *)malloc(M*sizeof(double));
    for (int i=0;i<M;i++){
		sum+=prob_original[i];
	}
	
    for (int i=0;i<M;i++){
		prob[i]=prob_original[i]/sum;
	}
	
    double *cum_prob = (double *)malloc((M+1)*sizeof(double));
    cum_prob[0]=0;
    for (int i=1;i<M;i++){
        cum_prob[i]=cum_prob[i-1]+prob[i-1];
    }
    cum_prob[M]=1;
    double u = runif();
    for (int i=0;i<M;i++){
        if ((u>=cum_prob[i])&&(u<=cum_prob[i+1])) {
            ans=i;
            break;
        }
    }
    
    free(prob);
    free(cum_prob);
    return ans;
}

void rDirich(double *alpha_prob,double *P_dirich, int M){ //sample from a Dirichlet distribution
    double *tmp, sum=0;
    tmp = (double *)malloc(M*sizeof(double));
    for (int i=0;i<M;i++){
        tmp[i] = rgamma(alpha_prob[i], 1);
        sum+=tmp[i];
    }
    for (int i=0;i<M;i++){
        P_dirich[i]=tmp[i]/sum;
    }
    free(tmp);
}


/*Main functions for implementing block Gibbs Sampler*/

/* sample from base distribution H */
void sample_H(int D, int p, double* par_0, double* par_1, double* par_2, double par_p_t, int ***L, double ***Lambda){
    /* D is the number of conditions, p is the node number,
     par_0 is the hyperparameter vector for the intensity parameter when there is no edge
     par_1 is the hyperparameter vector for the intensity parameter when there is an edge
     par_2 is the hyperparameter vector for the intensity parameter associated with a node
     par_p_t is a fixed hyperparameter indicating the probability of the presence of an edge */
    for(int d=0; d < D; d++){
        for(int i=0; i < p; i++){
            for(int j=i; j <p; j++){
                if(j==i){
                    Lambda[d][i][j]=rgamma(par_2[0], par_2[1]);
                    L[d][i][i]=-1;
                }else{
                    L[d][i][j]=rbernoulli(par_p_t);
                    if (L[d][i][j]==1){
                        Lambda[d][i][j]=rgamma(par_1[0],par_1[1]);
                    }else{
                        Lambda[d][i][j]=rgamma(par_0[0],par_0[1]);
                    }
                    L[d][j][i]=L[d][i][j];
                    Lambda[d][j][i]=Lambda[d][i][j];
                }
            }
        }
    }
}

/* conditional density of Y_t */
double log_dsty_Y(int D, int R, int p, int ****Y_t_n, double ***Lambda){
    /* D is the number of conditions, p is the node number,
     R is the replicate number, Y_t is the latent count data matrix, Lambda is the intensity parameter matrix*/
    double s=0, tmp;
    
    for (int d=0;d<D;d++){
        for (int i=0;i<p;i++){
            for (int j=i;j<p;j++){
                for (int r=0;r<R;r++){
                    tmp=log_dpoiss(Y_t_n[d][r][i][j],Lambda[d][i][j]);
                    s=s+tmp;
                }
            }
        }
    }
    return(s);
}


/* sample cla */
//find the maximum of a vector
double maximum(double* array,int size)
{
    double max;
    max=array[0];
    for (int i=1;i<size;i++){
        if (array[i]>max){
            max=array[i];
        }
    }
    return max;
}


//sample cla
int sample_cla(int M, int D, int p, int R, int n, double**** Lambda_t,double * P_dp_t,int *****Y_t){
    /* D is the number of conditions, p is the node number,
     R is the replicate number, M is the number of tables
     Lambda_t: M D p p; Y_t: N D R p p*/
    //update for sample n
    int tmp;
    double *ss, *ss_new, *Prob;
    int ****Y_t_n;
    Y_t_n = make4Darray_int_continuousMem(D, R, p, p);
    for(int d=0; d < D; d++){
        for(int r=0; r < R; r++){
            for(int i=0; i < p; i++){
                for(int j=0; j <p; j++){
                    Y_t_n[d][r][i][j] = Y_t[n][d][r][i][j] ;
                }
            }
        }
    }
    ss = (double *)malloc(M*sizeof(double));
    ss_new = (double *)malloc(M*sizeof(double));
    Prob = (double *)malloc(M*sizeof(double));
    double ss_max;
    for (int m=0;m<M;m++){
        ss[m]=log_dsty_Y(D, R, p, Y_t_n,Lambda_t[m]);
    }
    
    ss_max=maximum(ss,M);
    for (int m=0;m<M;m++){
        ss_new[m]=ss[m]-ss_max;
        Prob[m]=P_dp_t[m]*exp(ss_new[m]);
    }
    
    tmp=rdiscrete(M,Prob);
    free(ss);
    free(ss_new);
    free(Prob);
    delet4Darray_int_continuousMem(Y_t_n, D, R, p, p);
    return(tmp);
}



/* sample P_dp, the sample proportion of clusters, sum to be 1 */

void sample_P_dp(int N, int M, int * cla_t,double alpha,double *P_dp){
    int *M_arr;
    M_arr = (int *)malloc(M*sizeof(int));
    for(int m=0; m<M; m++){
        M_arr[m] = 0;
    }
    

    double *V_arr, sum;
    V_arr = (double *)malloc(M*sizeof(double));

    for (int m=0;m<=M-1;m++){
        for (int n=0;n<N;n++){
            if (cla_t[n]==m){
                M_arr[m]+=1;
            }
        }
    }
    
    for (int i=0;i<M-1;i++){
        sum=0.0;
        for (int j=i+1;j<M;j++){
            sum += M_arr[j];
        }
        V_arr[i]=rbeta(alpha/M+M_arr[i],alpha/M*(M-i-1)+sum);
    }
    P_dp[0]=V_arr[0];
    for(int i=1;i<M-1;i++){
        sum=0;
        for (int j=0;j<i;j++){
            sum+=P_dp[j];
        }
        P_dp[i]=(1-sum)*V_arr[i];
    }
    sum=0;
    for (int i=0;i<M-1;i++){
        sum += P_dp[i];
    }
    P_dp[M-1]=1-sum;
    free(M_arr);
    free(V_arr);
}

/* sample L_m for an occumpied table m*/
void sample_L_m(int D, int p, double ***Lambda_t,double par_p_t,double *par_0,double *par_1,int ***L_t){
    /* par_p_t is fixed */
    /* m is the table number */
    /* Lambda_t is the underlying p by p lambda matrix */
    double tmp[2],tmp_new[2],tmp_max,prob[2], sum;
    for (int d=0;d<D;d++){
        L_t[d][p-1][p-1]=-1;
        for (int i=0;i<p-1;i++){
            for (int j=i;j<p;j++){
                if(j==i){
                    L_t[d][i][j]=-1;
                }else{
                    tmp[0]=log_dgamma(Lambda_t[d][i][j], par_1[0], par_1[1]);
                    tmp[1]=log_dgamma(Lambda_t[d][i][j], par_0[0], par_0[1]);
                    if (isinf(tmp[0])&&isinf(tmp[1])){
                        tmp_new[0]=0.5;tmp_new[1]=0.5;
                    }else{
                        tmp_max=maximum(tmp,2);
                        tmp_new[0]=tmp[0]-tmp_max;
                        tmp_new[1]=tmp[1]-tmp_max;
                    }
                    prob[0]=par_p_t*exp(tmp_new[0]);
                    prob[1]=(1.0-par_p_t)*exp(tmp_new[1]);
                    sum=prob[0]+prob[1];
                    prob[0]=prob[0]/sum;
                    prob[1]=prob[1]/sum;
                    L_t[d][i][j]=rbernoulli(prob[0]);
                    L_t[d][j][i]=L_t[d][i][j];
                }
            }
        }
    }
    
}


void sample_Lambda_m(int D, int R, int N, int p, int *****Y_t, int ****L_t,int *cla_t,int m,double *par_0,double *par_1,double *par_2,double ***Lambda_t){
    /*Y_t: D R N p p; L_t: M D p p; Lambda_t: D p p */
    int total=0,tmp;
    int *ind;
    ind = (int *)malloc(N*sizeof(int));
    for(int i=0; i<N; i++){
        ind[i] = -1;
    }
    
    for (int i=0;i<N;i++){
        if(cla_t[i]==m){
            ind[total]=i;
            total+=1;
        }
    }
    
    for (int d=0;d<D;d++){
        for (int i=0;i<p;i++){
            for (int j=i;j<p;j++){
                tmp=0;
                for(int r =0; r<R; r++){
                    for (int k=0;k<total;k++){
                        tmp +=Y_t[ind[k]][d][r][i][j];
                    }
                }
                
                if (j==i){
                    Lambda_t[d][i][i]=rgamma(par_2[0]+tmp, par_2[1]+total*R);
                }else {
                    if (L_t[m][d][i][j]==1){
                        Lambda_t[d][i][j]=rgamma(par_1[0]+tmp, par_1[1]+total*R);
                    }else {
                        Lambda_t[d][i][j]=rgamma(par_0[0]+tmp, par_0[1]+total*R);
                    }
                    Lambda_t[d][j][i]=Lambda_t[d][i][j];
                }
            }
        }
    }
    free(ind);
}


/*update Y_t*/
/* proposal */
int Proposal(int p, int **Y_old,int **Y_star){  /*mark=0 means false;mark=1 means update */

    int mark=1;
    for (int i=0;i<p;i++){
        for (int j=0;j<p;j++){
            Y_star[i][j]=Y_old[i][j];
        }
    }
    
    int ind[2], t_ind;
    double *prob;
    prob = (double *)malloc(p*sizeof(double));
    for (int i=0;i<p;i++){
        prob[i]=1.0/p;
    }
    
    do{
        ind[0]=rdiscrete(p,prob);
        ind[1]=rdiscrete(p,prob);
    }while(ind[0]==ind[1]);
    
    if (ind[0]>ind[1]) {
        t_ind=ind[0];
        ind[0]=ind[1];
        ind[1]=t_ind;
    }
    
    double r;
    r=runif();
    if (r<=0.5){
        Y_star[ind[0]][ind[0]] = Y_star[ind[0]][ind[0]] - 1;
        Y_star[ind[0]][ind[1]] = Y_star[ind[0]][ind[1]] + 1;
        Y_star[ind[1]][ind[0]] = Y_star[ind[0]][ind[1]];
        Y_star[ind[1]][ind[1]] = Y_star[ind[1]][ind[1]] - 1;
    }else {
        Y_star[ind[0]][ind[0]] = Y_star[ind[0]][ind[0]] + 1;
        Y_star[ind[0]][ind[1]] = Y_star[ind[0]][ind[1]] - 1;
        Y_star[ind[1]][ind[0]] = Y_star[ind[0]][ind[1]];
        Y_star[ind[1]][ind[1]] = Y_star[ind[1]][ind[1]] + 1;
    }
    
    for (int i=0;i<2;i++){
        for (int j=0;j<2;j++){
            if (Y_star[ind[i]][ind[j]]<0) {mark=0;}
        }
    }
    free(prob);
    return mark;
}

//posterior density ratio
double post_dratio(int d,int r,int n, int p, int **Y_proposal,int *****Y_t,double ****Lambda_t,int cla){
    double s=0.0;
    for (int i=0;i<p;i++){
        for (int j=i;j<p;j++){
            s=s+log(Lambda_t[cla][d][i][j])*(Y_proposal[i][j]-Y_t[n][d][r][i][j])+logfactorial(Y_t[n][d][r][i][j])-logfactorial(Y_proposal[i][j]);
        }
    }
    return exp(s);
}

//sample the Y: p by p matrix
void sample_Y(int d,int r,int n,int p, double ****Lambda_t,int* cla_t,int *****Y_t){
    /*Y_t: N D R p p*/
    int mark=0;
    int **Y_proposal;
    Y_proposal = make2Darray_int_continuousMem(p,p);
    
    mark=Proposal(p, Y_t[n][d][r],Y_proposal);  /*mark=1 means false;mark=0 means update */
    
    int tmp;
    double ratio;

    if(mark==1){
        ratio=post_dratio(d,r,n,p,Y_proposal,Y_t,Lambda_t,cla_t[n]);
        if (ratio>1.0){
            ratio=1.0;
        }
        tmp=rbernoulli(ratio);
        if (tmp==1) {
            for (int i=0;i<p;i++){
                for (int j=0;j<p;j++){
                    Y_t[n][d][r][i][j] = Y_proposal[i][j];
                }
            }
        }
    }
    
    delet2Darray_int_continuousMem(Y_proposal, p, p);
}


int unique(int N, int *cla_t, int *cla_star){
    int mark,len;

    cla_star[0] = cla_t[0];
    len=1;
    for (int i=1;i<N;i++){
        mark=0;
        for (int j=0;j<len;j++){
            if (cla_star[j]==cla_t[i]){
                mark=1;
                break;
            }
        }
        if (mark==0) {
            cla_star[len]=cla_t[i];
            len+=1;
        }
    }
    return len;
}


