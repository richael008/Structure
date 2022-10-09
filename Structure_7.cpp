#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sstream>
#include <cstring>
#include <cmath>
#include <armadillo>
#include "mpi.h"

using std::cout;
using std::endl;
using std::string;


const double PI = 3.1415926535897932;
const double MaxGap =2.0;
const double StepSize=0.5;

const unsigned int TCut =101; 

string GenerateFileName();

double growth_prior(int node_depth, double gamma, double beta);

double update_sigma(const arma::vec& r, double sigma_hat, double sigma_old, double temperature);

arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision);

double ldcauchy(double x, double loc, double sig);

double cauchy_jacobian(double tau, double sigma_hat);

double expit(double x);

double activation(double x, double c, double tau);

void seq_gen_std2(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec);

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec);

bool Compare_Diff(double val, double cut, bool bcate);

void calculateOtherSideSuffStat(double * parent_suff_stat, double * lchild_suff_stat, double * rchild_suff_stat, bool compute_left_side);

void initialize_root_suffstat(arma::vec& Y_hat, double * suff_stat);

void updateNodeSuffStat(double * suff_stat, arma::vec& Y_hat, size_t row_ind);

double tau_proposal(double tau);

double log_tau_trans(double tau_new);

bool do_mh(double loglik_new, double loglik_old, double new_to_old, double old_to_new);

void calcSuffStat_categorical(double &temp_suff_stat,unsigned int * xorder,size_t start,size_t end,arma::vec& Y_hat);

double RandExp(const double width);

double rtnorm(double mean, double tau, double sd);

double logprior_tau(double tau, double tau_rate);

double r_exp(double tau_rate);

class F_OrderX
{
public:
	std::vector<size_t> variable_ind;
	std::vector<double> X_values;
	F_OrderX(unsigned int c);
	~F_OrderX();
};

struct Node;

class States
{
public:
	unsigned int p_categorical;
	unsigned int p_continuous;
	unsigned int p;
	unsigned int n;
	unsigned int n_test;
	unsigned int n_min;
	unsigned int max_depth;
	unsigned int MD;
	unsigned int n_cutpoints;
	//double tau;
	double sigma2;
	double sigma;
	double alpha;
	double beta;
	double no_split_penality ;
	double sigma_mu;
	double tau_rate;
	unsigned int RepeatT;
	unsigned int num_trees;
	unsigned int mtry;
	unsigned int num_sweeps;
	unsigned int num_burn;
	unsigned int num_save;
	double sigma_mu_hat;
	double sigma_hat;
	bool update_sigma_mu;
	bool update_s;
	bool update_alpha;
	bool update_beta;
	bool update_gamma;
	bool update_tau;
	bool update_tau_mean;
	double gamma;
	double shape;
	unsigned int np;
	double binaryOffset;
	bool binary;
	unsigned int temperature;
	double alpha_scale;
	double alpha_shape_1;
	double alpha_shape_2;
	double YMin;
	double YMax;
	bool verbose;
	double totalcount;
	bool Selected;                            //unused
	bool depthconstrain;                  //unused
	bool delhigest;                       //unused
	unsigned int widetype;
	unsigned int Try;
	bool mixrestart;
	unsigned int selectType;
	unsigned int winnercontrol;           //unused        
	unsigned int TreeCount[4];
	unsigned int Horder;
	double pmin;
	double resetP;

	int Level[3];


	arma::mat S_sigma;
	arma::mat S_sigma_mu;
	arma::mat y_hat_train;
	arma::mat y_hat_test;
	arma::mat y_hat_all;

	arma::vec split_count_all_tree;
	arma::vec split_count_all_tree_sum;
	arma::mat Node_counts;

	arma::vec split_count_current_tree;

	arma::mat change_histroy;
	arma::mat tau_histroy;
	arma::mat RecordTime;
	arma::mat split_Type;

	arma::mat Pages;
	arma::mat Relation;

	arma::mat PartialX;
	arma::vec Nstru;

	void UpdateSigmaMu(const arma::vec& means,double myrank);
	void UpdateSigma(const arma::vec& r,const double myrank);
	void UpdateSigmaA(const arma::vec& r,const double myrank);
  	void InitStates();
  	void ShowContents();
	int Findlocate(unsigned int ho);
	States();
	~States();
};

int Collect_Sample(double maxloglik,const double myrank, States * state);

class OrderX
{
public:
	unsigned int * Xorder;
	std::vector<unsigned int> Xcounts;
	std::vector<unsigned int> X_num_unique;
	size_t N;

	OrderX(const arma::mat & x, F_OrderX * f_orderx, States * states);
	OrderX(size_t n, size_t p, unsigned int uc, unsigned int cc);
	void Split(OrderX * Lorder, OrderX * Rorder, arma::vec& Y_hat, const arma::mat & x, F_OrderX * f_orderx, States * states, size_t split_var, size_t split_point, Node * current_node);
	~OrderX();
};

struct Node {
	bool is_leaf;
	bool is_root;
	Node* left;
	Node* right;
	Node* parent;
	unsigned int var;
	double val;
	double tau;
	unsigned int nid;
	double mu;
	double current_weight;
	//double loglike_node;
	double suff_stat[3];
	unsigned int depth;

	void updateSplit(States * state);
	void UpdateMuA(double * ML,int updatecount, std::vector<double> & means);

	// ,unsigned int treeid,double myrank

	void grow_from_root(States * state,arma::vec& Y_hat,const arma::mat & x,F_OrderX * f_orderx,OrderX * orderx,unsigned int sweeps,unsigned int norder);
	double updateWidth(const arma::vec& Y,const arma::mat& X,States * state);
	double updateWidth_Opt(const arma::vec& Y,const arma::mat& X,States * state);
  	void UpdateMu(const arma::vec& Y, const arma::mat& X, States *  state, std::vector<double> & means);
  	void getnodes(std::vector<Node *> &v);
  	Node * getptr(const unsigned int Tid);
  	void GetW(const arma::mat& X, int i,States * state);
	unsigned int height();  
  	void SetTau(double tau_new);
  	void AddLeaves();
	size_t nbots();
  	void Root(void);
	Node();
	~Node();
};

double tree_loglik(Node* node, int node_depth, double gamma, double beta);

double LogLT(Node* n, const arma::vec& Y, const arma::mat& X, States *state);

double LogLTA(Node* n, const arma::vec& Y,const arma::mat& X, States *state,const double myrank,int Mucount,double * Mulist );

double LogLTB(Node* n, const arma::vec& Y,const arma::mat& X, States *state,int Mucount,double * Mulist );

void leaves(Node* x, std::vector<Node*>& leafs);

std::vector<Node*> leaves(Node* x);

void calculate_loglikelihood_continuous(size_t cutcount,std::vector<double> &loglike,const std::vector<size_t> &subset_vars,States * state,OrderX * orderx,arma::vec& Y_hat,const arma::mat & x,double &loglike_max,	Node *tree_pointer);

void calculate_loglikelihood_categorical(std::vector<double> &loglike,size_t &loglike_start,const std::vector<size_t> &subset_vars,States * state,F_OrderX * f_orderx,OrderX * orderx,arma::vec& Y_hat,const arma::mat & x,double &loglike_max,Node *tree_pointer);
//,unsigned int sweeps ,unsigned int norder ,unsigned int treeid ,double myrank
void BART_likelihood_all(States * state,arma::vec& Y_hat,const arma::mat & x,F_OrderX * f_orderx,OrderX * orderx,const std::vector<size_t> &subset_vars,size_t &split_var,size_t &split_point,Node *tree_pointer,bool &no_split,size_t & split_count);

void calcSuffStat_continuous(double &temp_suff_stat,unsigned int * xorder,std::vector<size_t> &candidate_index,size_t index,arma::vec& Y_hat);
//,unsigned int sweeps ,unsigned int norder ,unsigned int treeid ,double myrank
void calculate_likelihood_no_split(std::vector<double> &loglike,double &loglike_max,States * state,Node *tree_pointer);

double likelihood(double & temp_suff_stat,double * suff_stat_all,size_t N_left,bool left_side,bool no_split,States *state);

double likelihood(double * suff_stat_all,States *state);

void GetSuffStats(Node* n,const arma::vec& y,const arma::mat& X,States *  state,arma::vec& mu_hat_out,arma::mat& Omega_inv_out);

void GetSuffStats(Node* n, const arma::vec& y,const arma::mat& X, States *  state,arma::vec& mu_hat_out, arma::mat& Omega_inv_out,const double myrank);

std::ostream &operator<<(std::ostream &os, Node *t);

arma::vec predict(Node* n, const arma::mat& X, States * state);

std::vector<Node*> init_forest(States * state);

void incSuffStat(arma::vec& Y_hat,size_t index_next_obs,double &suffstats);











string GenerateFileName()
{
 string szRet = "";
 char timeBuffer[30];
 time_t nowtime = time(NULL);
 struct tm *timeTemp;
 timeTemp=localtime(&nowtime);
 sprintf(timeBuffer, "RTime-%04d-%02d-%02d_%02d-%02d-%02d.csv", timeTemp->tm_year + 1900, timeTemp->tm_mon + 1,
 timeTemp->tm_mday, timeTemp->tm_hour, timeTemp->tm_min, timeTemp->tm_sec);
 szRet = timeBuffer;
 return szRet;
}



double growth_prior(int node_depth, double gamma, double beta)
{
  return gamma * pow(1.0 + node_depth, -beta);
}


int Collect_Sample(double maxloglik,const double myrank, States * state)
{
	//std::stringstream srnstr; 
	double send_buffer;
    double *recv_buffer;
	arma::vec lik = arma::zeros<arma::vec>(state->np+1);
	double maxlik;
	int ind=0;

    recv_buffer = new double[state->np+1];	
	send_buffer=maxloglik;
    MPI_Gather(&send_buffer, 1, MPI_DOUBLE,  recv_buffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myrank == 0)
    {
		
        for (unsigned int i = 0; i <= state->np; i++)
		{
			lik(i)=recv_buffer[i];
		}
		
		maxlik=lik.max();
        for (unsigned int i = 0; i <= state->np; i++)
		{		
			lik(i)=exp(lik(i)-maxlik);
		}
		double Urand=arma::randu()*sum(lik);
		double acum= 0.0;

        if (state->selectType==3)
		{
		    arma::uvec indices = arma::sort_index(lik);
            ind  = indices[(state->np+1)/2];  
		}
		else
		{

			for (unsigned int i = 0; i <= state->np; i++)
			{   
				acum += lik(i);
				if (acum >= Urand)
				{
					ind = i;
					break;
				}
			}



		}
	}

	MPI_Bcast(&ind,1,MPI_INT,0,MPI_COMM_WORLD);
    delete [] recv_buffer;
	return ind;
}


double update_sigma(const arma::vec& r, double sigma_hat, double sigma_old, double temperature) 
{
	double SSE = dot(r, r) * temperature;
	double n = r.size() * temperature;
	double shape = 0.5 * n + 1.0;
	double scale = 2.0 / SSE;
	arma::vec v = arma::randg<arma::vec>(1, arma::distr_param(shape, scale));
	double sigma_prop = pow(v(0), -0.5);
	double tau_prop = pow(sigma_prop, -2.0);
	double tau_old = pow(sigma_old, -2.0);
	double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) -	cauchy_jacobian(tau_old, sigma_hat);
	return log(arma::randu()) < loglik_rat ? sigma_prop : sigma_old;
}


// arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision) 
// {
// 	arma::vec z = arma::randn<arma::vec>(mean.size());
// 	arma::mat Sigma = inv_sympd(Precision);
// 	arma::mat L = chol(Sigma, "lower");
// 	arma::vec h = mean + L * z;
// 	return h;
// }


arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision) 
{

	arma::mat Sigma = inv_sympd(Precision);
    arma::vec h = arma::mvnrnd(mean, Sigma);
	return h;
}


double expit(double x) 
{
	return 1.0 / (1.0 + exp(-x));
}


double activation(double x, double c, double tau) {
	return 1.0 - expit((x - c) / tau);
}


bool do_mh(double loglik_new, double loglik_old, double new_to_old, double old_to_new)
{
	double cutoff = loglik_new + new_to_old - loglik_old - old_to_new;
	return log(arma::randu()) < cutoff ? true : false;
}


void seq_gen_std2(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
	vec[0] = 0;
	if (length_out == 1)
	{
		vec[1] = (end + start) / 2;
		vec[2] = end+1;
	}
	else
	{
		double incr = (double)(end - start) / (double)(length_out - 1);
		for (size_t i = 1; i < length_out + 1; i++)
		{
			vec[i] = (size_t)(incr * (i - 1)) + start;
		}
		vec[length_out + 1] = end + 1;

	}
	return;
}


void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
	vec[0] = start;

	double incr = (double)(end - start) / (double)(length_out - 1);
	for (size_t i = 1; i < length_out ; i++)
	{
		vec[i] = (size_t)(incr * i) + start;
	}
	return;
}


double tau_proposal(double tau) {
	double U = 2.0 * arma::randu() - 1;
	return pow(5.0, U) * tau;
}


// double logprior_tau(double tau, double tau_rate)
// {
// 	return ldexp(tau, tau_rate);
// }

 double logprior_tau(double tau, double tau_rate)
 {
 	return -1*tau_rate*tau;
 }


double r_exp(double tau_rate)
{
	return -1.0/tau_rate*log(arma::randu());
}


double log_tau_trans(double tau_new) {
	return -log(tau_new);
}


bool Compare_Diff(double val, double cut, bool bcate)
{
	if (bcate)
	{
		return  (val == cut) ? true : false;
	}
	else
	{
		return  (val <= cut) ? true : false;
	}
}


void calculateOtherSideSuffStat(double * parent_suff_stat, double * lchild_suff_stat, double * rchild_suff_stat, bool compute_left_side)
{

	if (compute_left_side)
	{
		for (size_t i = 0; i < 3; i++)
		{
			rchild_suff_stat[i] = parent_suff_stat[i] - lchild_suff_stat[i];
		}

	}
	else
	{
		for (size_t i = 0; i < 3; i++)
		{
			lchild_suff_stat[i] = parent_suff_stat[i] - rchild_suff_stat[i];
		}
	}
	return;
}


void initialize_root_suffstat(arma::vec& Y_hat, double * suff_stat)
{
	suff_stat[0] = arma::sum(Y_hat);
	suff_stat[1] = arma::sum(Y_hat%Y_hat);
	suff_stat[2] = Y_hat.n_elem;
	return;
}


void updateNodeSuffStat(double * suff_stat, arma::vec& Y_hat, size_t row_ind)
{
	suff_stat[0] += Y_hat[row_ind];
	suff_stat[1] += pow(Y_hat[row_ind], 2);
	suff_stat[2] += 1;
	return;
}


double RandExp(const double width)
{
  return (-1*log(arma::randu())*width) ;
}


double rtnorm(double mean, double tau, double sd)
{
  double x, z, lambda;
  /* Christian Robert's way */
  //assert(mean < tau); //assert unnecessary: Rodney's way
  tau = (tau - mean)/sd;
  /* originally, the function did not draw this case */
  /* added case: Rodney's way */
  if(tau<=0.) {
    /* draw until we get one in the right half-plane */
    do { z=arma::randn(); } while (z < tau);
  }
  else {
    /* optimal exponential rate parameter */
    lambda = 0.5*(tau + sqrt(tau*tau + 4.0));

    /* do the rejection sampling */
    do {
      z = RandExp(1.0)/lambda + tau;
      //z = lambda*gen.exp() + tau;
    } while (arma::randu() > exp(-0.5*pow(z - lambda, 2.)));
  }
  /* put x back on the right scale */
  x = z*sd + mean;
  //assert(x > 0); //assert unnecessary: Rodney's way
  return(x);
}


F_OrderX::F_OrderX(unsigned int c)
{
	variable_ind = std::vector<size_t>(c);
	X_values = std::vector<double>(0);
}


States::States()
{

}


void States::InitStates()
{   
	
	if (Horder==1)
	{
		this->split_count_all_tree= arma::zeros<arma::vec>(p);
		this->split_count_all_tree_sum= arma::zeros<arma::vec>(p);
	}
	else if (Horder==2)
	{
		this->split_count_all_tree= arma::zeros<arma::vec>(p+(p*(p-1)/2));
		this->split_count_all_tree_sum= arma::zeros<arma::vec>(p+(p*(p-1)/2));
	}
	else if (Horder==3)
	{
		this->split_count_all_tree= arma::zeros<arma::vec>(p+(p*(p-1)/2)+(p*(p-1)*(p-2)/6));
		this->split_count_all_tree_sum= arma::zeros<arma::vec>(p+(p*(p-1)/2)+(p*(p-1)*(p-2)/6));
	}
	
	this->split_count_current_tree = arma::zeros<arma::vec>(p);
	this->Node_counts=arma::ones<arma::mat>(num_sweeps, num_trees)*max_depth;
	this->MD=this->max_depth;

	change_histroy=arma::ones<arma::mat>(num_sweeps, num_trees);
	tau_histroy=arma::zeros<arma::mat>(num_sweeps, num_trees);
	this->S_sigma = arma::zeros<arma::mat>(num_sweeps -num_burn, this->num_trees);
	this->S_sigma_mu = arma::zeros<arma::mat>(num_sweeps -num_burn,Horder);
	this->y_hat_train = arma::zeros<arma::mat>(this->num_save, n);
	this->y_hat_test = arma::zeros<arma::mat>(this->num_save, n_test);
	this->y_hat_all =arma::zeros<arma::mat>(this->n, this->num_trees);
	this->RecordTime = arma::zeros<arma::mat>(num_sweeps,num_trees*4);

    this->split_Type = arma::zeros<arma::mat>(num_sweeps, num_trees);
	this->Level[0]=0;
	this->Level[1]=0;
	this->Level[2]=0;
	


}


void States::ShowContents()
{
	std::cout<< this->p_categorical << "\t p_categorical"<<std::endl;
	std::cout<< this->p_continuous << "\t\t p_continuous"<<std::endl;
	std::cout<< this->p << "\t\t p"<<std::endl;
	std::cout<< this->n << "\t\t n"<<std::endl;
	std::cout<< this->n_test << "\t\t n_test"<<std::endl;
	std::cout<< this->n_min << "\t\t n_min"<<std::endl;
	std::cout<< this->max_depth << "\t\t max_depth"<<std::endl;
	std::cout<< this->n_cutpoints << "\t\t n_cutpoints"<<std::endl;
	//std::cout<< this->tau << "\t\t tau"<<std::endl;
	std::cout<< this->sigma2 << "\t\t sigma2"<<std::endl;
	std::cout<< this->sigma << "\t\t sigma"<<std::endl;
	std::cout<< this->alpha << "\t\t alpha"<<std::endl;
	std::cout<< this->beta << "\t\t beta"<<std::endl;
	std::cout<< this->no_split_penality << "\t\t no_split_penality"<<std::endl;
	std::cout<< this->sigma_mu << "\t\t sigma_mu"<<std::endl;
	std::cout<< this->tau_rate << "\t\t tau_rate"<<std::endl;
	std::cout<< this->RepeatT << "\t\t RepeatT"<<std::endl;
	std::cout<< this->num_trees << "\t\t num_trees"<<std::endl;
	std::cout<< this->mtry << "\t\t mtry"<<std::endl;
	std::cout<< this->num_sweeps << "\t\t num_sweeps"<<std::endl;
	std::cout<< this->num_burn << "\t\t num_burn"<<std::endl;
	std::cout<< this->num_save << "\t\t num_save"<<std::endl;
	std::cout<< this->sigma_mu_hat << "\t\t sigma_mu_hat"<<std::endl;
	std::cout<< this->sigma_hat << "\t\t sigma_hat"<<std::endl;
	std::cout<< this->update_sigma_mu << "\t\t update_sigma_mu"<<std::endl;
	std::cout<< this->update_s << "\t\t update_s"<<std::endl;
	std::cout<< this->update_alpha << "\t\t update_alpha"<<std::endl;
	std::cout<< this->update_beta << "\t\t update_beta"<<std::endl;
	std::cout<< this->update_gamma << "\t\t update_gamma"<<std::endl;
	std::cout<< this->update_tau << "\t\t update_tau"<<std::endl;
	std::cout<< this->update_tau_mean << "\t\t update_tau_mean"<<std::endl;
	std::cout<< this->gamma << "\t\t gamma"<<std::endl;
	std::cout<< this->shape << "\t\t shape"<<std::endl;
	std::cout<< this->np << "\t\t np"<<std::endl;
	std::cout<< this->binaryOffset << "\t\t binaryOffset"<<std::endl;
	std::cout<< this->binary << "\t\t binary"<<std::endl;
	std::cout<< this->temperature << "\t\t temperature"<<std::endl;
	std::cout<< this->alpha_scale << "\t\t alpha_scale"<<std::endl;
	std::cout<< this->alpha_shape_1 << "\t\t alpha_shape_1"<<std::endl;
	std::cout<< this->alpha_shape_2 << "\t\t alpha_shape_2"<<std::endl;
	std::cout<< this->YMin << "\t\t YMin"<<std::endl;
	std::cout<< this->YMax << "\t\t YMax"<<std::endl;
	std::cout<< this->delhigest << "\t\t delhigest"<<std::endl;
	std::cout<< this->depthconstrain << "\t\t depthconstrain"<<std::endl;
	std::cout<< this->verbose << "\t\t verbose"<<std::endl;
	std::cout<< this->Selected << "\t\t Selected"<<std::endl;
	std::cout<< this->widetype << "\t\t widetype"<<std::endl;
	std::cout<< this->Try << "\t\t Try"<<std::endl;	
	std::cout<< this->mixrestart << "\t\t mixrestart"<<std::endl;
	std::cout<< this->selectType << "\t\t selectType"<<std::endl;
	std::cout<< this->winnercontrol << "\t\t winnercontrol"<<std::endl;


	

}

int States::Findlocate(unsigned int h)
{
	unsigned int indx[3];

	bool MFail=false;
	unsigned int j=0;
	for (unsigned int i=0;i<this->p;i++)
	{
		if (this->split_count_current_tree[i])
		{
			if (j<=h) 
			{
				indx[j++]=i;
			}
			else
			{
				MFail=true;
				break;
			}
		}
	}
	if (j != h+1)
	{
		MFail=true;
	}

	if (MFail)
	{
		return -1;
	}
	else
	{
		if (h==0)
		{
			return indx[0];	
		}
		else if (h==1)
		{
			return  indx[0]*this->p +indx[1]+this->p-(indx[0]+1)*(indx[0]+2)/2;
		}
		else
		{
			return -1;
		}
	}
}

void States::UpdateSigmaMu(const arma::vec& means,double myrank) 
{
	double T_sigma_mu=0;
	if(myrank==0)
	{
		T_sigma_mu = update_sigma(means, sigma_mu_hat, sigma_mu, 1.0);
	}
	MPI_Bcast(&T_sigma_mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	sigma_mu=T_sigma_mu;
}

void States::UpdateSigma(const arma::vec& r,const double myrank) 
{

	double *SSEN = new double[2];
	double *SSEND = new double[2];
	double T_sigma=0;

	SSEN[0] = dot(r,r) ;
	SSEN[1] = r.size() ;
	SSEND[0]=0;
	SSEND[1]=0;
	MPI_Reduce(SSEN,SSEND,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);


	if (myrank==0)
	{  
		double shape = 0.5 * SSEND[1] + 1.0;
		double scale = 2.0 / SSEND[0];
		arma::vec v = arma::randg<arma::vec>(1, arma::distr_param(shape,scale));
		double sigma_prop = pow(v(0), -0.5);

		double tau_prop = pow(sigma_prop, -2.0);
		double tau_old = pow(sigma, -2.0);

		double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) - cauchy_jacobian(tau_old, sigma_hat);

		T_sigma = (log(arma::randu()) < loglik_rat ? sigma_prop : sigma);
	}
	MPI_Bcast(&T_sigma,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	sigma=T_sigma;
	sigma2=sigma*sigma;

	delete[] SSEN;
	delete[] SSEND;
}

void States::UpdateSigmaA(const arma::vec& r,const double myrank) 
{


	double *SSEND = new double[2];
	double T_sigma=0;

	if (myrank==0)
	{  
		SSEND[0] = dot(r,r) ;
		SSEND[1] = r.size() ;
		double shape = 0.5 * SSEND[1] + 1.0;
		double scale = 2.0 / SSEND[0];
		arma::vec v = arma::randg<arma::vec>(1, arma::distr_param(shape,scale));
		double sigma_prop = pow(v(0), -0.5);

		double tau_prop = pow(sigma_prop, -2.0);
		double tau_old = pow(sigma, -2.0);

		double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) - cauchy_jacobian(tau_old, sigma_hat);

		T_sigma = (log(arma::randu()) < loglik_rat ? sigma_prop : sigma);
	}
	MPI_Bcast(&T_sigma,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	sigma=T_sigma;
	sigma2=sigma*sigma;
	delete[] SSEND;
}



OrderX::OrderX(const arma::mat & x, F_OrderX * f_orderx, States * states)
{
	Xorder = new unsigned int[x.n_rows *x.n_cols];
   	Xcounts = std::vector<unsigned int>(0);
	X_num_unique = std::vector<unsigned int>(0);

	N = x.n_rows;
	for (unsigned int i = 0; i < x.n_cols; i++)
	{
		arma::uvec indices = arma::sort_index(x.col(i));
		for (unsigned int j = 0; j < N; j++)
		{
			Xorder[i*N + j] = indices[j];
		}
	}
	unsigned int count_unique = 0;
	double current_value = 0.0;
	unsigned int total_points = 0;
	f_orderx->variable_ind[0] = 0;
	for (unsigned int i = states->p_continuous; i < states->p; i++)
	{
		Xcounts.push_back(1);
		current_value = x(Xorder[i*N], i);
		f_orderx->X_values.push_back(current_value);
		count_unique = 1;

		for (unsigned int j = 1; j < N; j++)
		{
			if (x(Xorder[i*N + j], i) == current_value)
			{
				Xcounts[total_points]++;
			}
			else
			{
				current_value = x(Xorder[i*N + j], i);
				f_orderx->X_values.push_back(current_value);
				Xcounts.push_back(1);
				count_unique++;
				total_points++;
			}
		}
		f_orderx->variable_ind[i + 1 - states->p_continuous] = count_unique + f_orderx->variable_ind[i - states->p_continuous];
		X_num_unique.push_back(count_unique);
		total_points++;
	}
}

OrderX::OrderX(size_t n, size_t p, unsigned int uc, unsigned int cc)
{
	Xorder = new unsigned int[n*p];
	Xcounts = std::vector<unsigned int>(cc);
	X_num_unique = std::vector<unsigned int>(uc);
	N = n;
}

OrderX::~OrderX()
{
	delete[] Xorder;
}

void OrderX::Split(OrderX * Lorder, OrderX * Rorder, arma::vec& Y_hat, const arma::mat & x, F_OrderX * f_orderx, States * states, size_t split_var, size_t split_point, Node * current_node)
{
	bool compute_left_side = (Lorder->N <= Rorder->N);
	bool bcatg = (split_var >= states->p_continuous);
	double cutvalue = x(Xorder[split_var*N + split_point], split_var);

	if (split_var < states->p_continuous)
	{
		if (compute_left_side)
		{
			for (size_t j = 0; j <= split_point; j++)
			{
				updateNodeSuffStat(current_node->left->suff_stat, Y_hat, Xorder[split_var*N + j]);
			}
		}
		else
		{
			for (size_t j = split_point + 1; j <N; j++)
			{
				updateNodeSuffStat(current_node->right->suff_stat, Y_hat, Xorder[split_var*N + j]);
			}
		}
	}
	else
	{
		if (compute_left_side)
		{

			for (size_t j = split_point; j >= 0 && (x(Xorder[split_var*N + j], split_var) == cutvalue); j--)
			{
				updateNodeSuffStat(current_node->left->suff_stat, Y_hat, Xorder[split_var*N + j]);
			}


		}
		else
		{

			for (size_t j = 0; j <N; j++)
			{
				if (x(Xorder[split_var*N + j], split_var) != cutvalue)
				{
					updateNodeSuffStat(current_node->right->suff_stat, Y_hat, Xorder[split_var*N + j]);
				}
			}
		}
	}
	calculateOtherSideSuffStat(current_node->suff_stat, current_node->left->suff_stat, current_node->right->suff_stat, compute_left_side);

	size_t start=0;
	size_t end=0;
	bool SoloCount;
	size_t X_counts_index=0;
	for (size_t i = 0; i < states->p; i++)
	{
		size_t left_ix = 0;
		size_t right_ix = 0;

		if (i >= states->p_continuous)
		{
			start = f_orderx->variable_ind[i - states->p_continuous];
			end = f_orderx->variable_ind[i + 1 - states->p_continuous];
			X_counts_index = start;
		}
		SoloCount = ((i >= states->p_continuous) && (i == split_var));


		if (SoloCount)
		{
			for (size_t k = start; k < end; k++)
			{
				if (f_orderx->X_values[k] == cutvalue)
				{
					Lorder->Xcounts[k] = Xcounts[k];
				}
				else
				{
					Rorder->Xcounts[k] = Xcounts[k];
				}
			}
		}

		for (size_t j = 0; j < N; j++)
		{
			if (i >= states->p_continuous)
			{
				while (x(Xorder[i*N + j], i) != f_orderx->X_values[X_counts_index])
				{
					X_counts_index++;
				}
			}

			if (Compare_Diff(x(Xorder[i*N + j], split_var), cutvalue, bcatg))
			{
				Lorder->Xorder[i*Lorder->N + left_ix] = Xorder[i*N + j];
				left_ix = left_ix + 1;

				if ((i >= states->p_continuous) &  (!(SoloCount)))
				{
					Lorder->Xcounts[X_counts_index]++;
				}

			}
			else
			{
				Rorder->Xorder[i*Rorder->N + right_ix] = Xorder[i*N + j];
				right_ix = right_ix + 1;

				if ((i >= states->p_continuous) &  (!(SoloCount)))
				{
					Rorder->Xcounts[X_counts_index]++;
				}


			}
		}



		if (i >= states->p_continuous)
		{
			for (size_t j = start; j < end; j++)
			{
				if (Lorder->Xcounts[j] > 0)
				{
					Lorder->X_num_unique[i - states->p_continuous] = Lorder->X_num_unique[i - states->p_continuous] + 1;
				}
				if (Rorder->Xcounts[j] > 0)
				{
					Rorder->X_num_unique[i - states->p_continuous] = Rorder->X_num_unique[i - states->p_continuous] + 1;
				}
			}
		}
	}
}


size_t Node::nbots()
{
    if (left == 0)
    { 
        return 1;
    }
    else
    {
        return left->nbots() + right->nbots();
    }
}


void Node::updateSplit(States * state)
{
    if (left == 0)
    { 
        return ;
    }
    else
    {
		state->split_count_current_tree[var]=1;
        left->updateSplit(state);
		right->updateSplit(state);
		return ;
    }
}


void Node::getnodes(std::vector<Node *> &v)
{
	v.push_back(this);
	if (!is_leaf)
	{
		left->getnodes(v);
		right->getnodes(v);
	}
}


unsigned int Node::height()
{
	if (left==0)
	{
		return depth;
	}
	else
	{
		unsigned int lH=left->height();
		unsigned int rH=right->height();
		return( lH>rH?lH:rH );
	}
}

Node * Node::getptr(const unsigned int Tid)
{
	if (this->nid == Tid) return this;
	if (is_leaf) return 0;
	if (left)
	{
		Node * lp = left->getptr(Tid);
		if (lp) return lp;
	}
    if (right)
	{
		Node * rp = right->getptr(Tid);
		if (rp) return rp;
	}

	return 0;
}


void Node::SetTau(double tau_new) 
{
	tau = tau_new;
	if (!is_leaf) {
		left->SetTau(tau_new);
		right->SetTau(tau_new);
	}
}


void Node::AddLeaves() {
	left = new Node;
	right = new Node;
	is_leaf = false;
	left->parent = this;
	right->parent = this;
	left->nid = nid * 2;
	right->nid = nid * 2 + 1;
	left->depth = depth + 1;
	right->depth = depth + 1;
}


void Node::Root(void)
{
	is_root = true;
	parent = 0;
	nid = 1;
	depth = 0;
	current_weight = 1.0;
}


Node::Node() 
{
	is_leaf = true;
	is_root = false;
	left = 0;
	right = 0;
	parent = 0;
	var = 0;
	val = 0.0;
	tau=0;
	nid=0;
	mu = 0.0;
	current_weight = 0.0;
	//loglike_node=0.0;
	suff_stat[0]=0.0;
	suff_stat[1]=0.0;
	suff_stat[2]=0.0;
	depth=0;
}


Node::~Node() {
	if (!is_leaf) {
		delete left;
		delete right;
	}
}


void Node::GetW(const arma::mat& X, int i,States * state) 
{

	if (!is_leaf) {
		if (current_weight == 0)
		{
			left->current_weight = 0;
			right->current_weight = 0;
		}
		else if (var < state->p_continuous)
		{
			double weight;

			if (tau==0)
			{
				weight =  (X(i, var) <= val ? 1 : 0);
			}
			else
			{
				weight = activation(X(i, var), val, tau);
			} 

			left->current_weight = weight * current_weight;
			right->current_weight = (1 - weight) * current_weight;
		}
		else 
		{
			double weight = (X(i, var)==val ?1:0) ;
			left->current_weight = weight * current_weight;
			right->current_weight = (1 - weight) * current_weight;
		}

		left->GetW(X, i,state);
		right->GetW(X, i,state);

	}
}


void Node::UpdateMu(const arma::vec& Y, const arma::mat& X, States *  state, std::vector<double> & means) 
{

	std::vector<Node*> leafs = leaves(this);
	int num_leaves = leafs.size();


	arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
	arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
	GetSuffStats(this, Y, X, state, mu_hat, Omega_inv);

	arma::vec mu_samp = rmvnorm(mu_hat, Omega_inv);
	for (int i = 0; i < num_leaves; i++) {
		leafs[i]->mu = mu_samp(i);
		means.push_back(mu_samp(i));
	}
}


void Node::UpdateMuA(double * ML,int updatecount, std::vector<double> & means) 
{
  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();
  if (num_leaves!=updatecount)
  {
	cout<<"num of leaves not match"<<endl;
  }
  else
  {
	for(int i = 0; i < num_leaves; i++) 
	{
		leafs[i]->mu = ML[i];
		means.push_back(ML[i]);
	}
  } 
}

//,unsigned int treeid ,double myrank

void Node::grow_from_root(  States * state, 	arma::vec& Y_hat,	const arma::mat & x,	F_OrderX * f_orderx,	OrderX * orderx,  unsigned int sweeps ,unsigned int norder )
{	
	size_t N_Xorder = orderx->N;
	size_t p = state->p;
	size_t split_var;
	size_t split_point;
	size_t split_count=0;
	bool no_split = false;

	if (N_Xorder < 2*(state->n_min))
	{
		no_split = true;
  	}
	if (this->depth >= state->max_depth )
	{
		no_split = true;
	}


	if (this->depth==0)
	{

		state->Nstru = arma::zeros<arma::vec>(norder+1);

		if (state->Selected)
		{
			unsigned int Selecti;
			if ((state->Level[norder+1]-state->Level[norder])==1)
			{
			   Selecti=state->Level[norder];
			}
			else
			{
			   Selecti=arma::randi(arma::distr_param(state->Level[norder], state->Level[norder+1]-1));
			}


			if (norder==0)
			{
				state->Nstru[0]=state->Relation(Selecti,1);
			}
			else if (norder==1)
			{
				arma::vec BShuffle=arma::zeros<arma::vec>(2);
				BShuffle[0]=state->Relation(Selecti,1);
				BShuffle[1]=state->Relation(Selecti,2);
				arma::vec AShuffle=arma::shuffle(BShuffle);
				state->Nstru[0]=AShuffle[0];
				state->Nstru[1]=AShuffle[1];
			}
			else
			{
				exit(1);
			}
		}
		else
		{
			if (norder>0)
			{
				arma::vec q(p, arma::fill::randu);
				arma::uvec indices = arma::sort_index(q);

				for (unsigned int i = 0; i < norder; i++)
				{
					state->Nstru[i]=indices[i];
				}
			}
			state->Nstru[norder]=-1;
		}	

	}

	std::vector<size_t> subset_vars(0);

	bool bselect;
	bselect=true;

	if (this->depth<norder)
	{
		split_var=state->Nstru[this->depth];
		double split_p;
		split_p=state->pmin+(1-2*state->pmin)*arma::randu();
        split_point=(int) (split_p*N_Xorder);

		double TempVar=x(orderx->Xorder[split_var*N_Xorder + split_point], split_var);
		while ((split_point < N_Xorder - 1) && (x(orderx->Xorder[split_var*N_Xorder + split_point+1], split_var) == TempVar))
		{
			split_point = split_point + 1;
		}
		split_count=  split_point + 1;
		bselect=false;
	}
	else if ( (this->depth==norder) && (state->Nstru[norder]==-1))
	{
		subset_vars.resize(p-norder);
		for (unsigned int i = 0,ins_p=0; i < p; i++)
		{
			bool bOccur=true;
		    for (unsigned int j = 0; j < norder; j++)
		   {
			   if (state->Nstru[j]==i)
			   {
				   bOccur=false;
				   break;
			   }
		   }   
		   if (bOccur) 
		   {
				subset_vars[ins_p++] = i;
		   }
		}
	}
	else if ( (this->depth==norder) && (state->Nstru[norder] != -1))
	{
		subset_vars.resize(1);
		subset_vars[0] =state->Nstru[norder];
	}
	else
	{
		subset_vars.resize(norder+1);
		for (unsigned int i = 0; i <= norder; i++)
		{
			subset_vars[i]=	state->Nstru[i];
		}			
		
		
		// for (unsigned int i = 0, ins_p=0; i < p; i++)
		// {
		// 	bool bOccur=false;
		//     for (unsigned int j = 0; j <= norder; j++)
		//    {
		// 	   if (state->Nstru[j]==i)
		// 	   {
		// 		   bOccur=true;
		// 		   break;
		// 	   }
		//    }   
		//    if (bOccur) 
		//    {
		// 		subset_vars[ins_p++] = i;
		//    }
		// }
	}

	if ((!no_split) && (bselect))
	{
		//,sweeps,norder,treeid,myrank
		BART_likelihood_all(state, Y_hat, x, f_orderx, orderx, subset_vars, split_var, split_point, this, no_split,split_count);
		//this->loglike_node = likelihood(this->suff_stat,  state);
	}

	if (no_split )
	{
		return;
	}

	this->var = split_var;
	this->val = x(orderx->Xorder[split_var*N_Xorder + split_point], split_var); 

	if ( (this->depth==norder) && (state->Nstru[norder]==-1) )
	{
		state->Nstru[norder]=split_var;
	}

	if ((split_count <state->n_min ) ||   ( split_count>  N_Xorder- state->n_min))
	{
		cout<<this->var<<" " <<this->val<<" Something   split_count <state->n_min  split_count>  N_Xorder- state->n_min   wrong,stopped"<<endl;
		exit(1);
	}

	AddLeaves();

	OrderX * lorderx = new OrderX(split_count, state->p, (unsigned int)orderx->X_num_unique.size(), (unsigned int)orderx->Xcounts.size());
	OrderX * rorderx = new OrderX(orderx->N - split_count, state->p, (unsigned int)orderx->X_num_unique.size(), (unsigned int)orderx->Xcounts.size());


	orderx->Split(lorderx, rorderx, Y_hat, x, f_orderx, state, split_var, split_point, this);


    if (arma::randu()<0.5)
	{
		//,treeid,myrank
		this->left->grow_from_root(state, Y_hat, x, f_orderx, lorderx,sweeps,norder);
		this->right->grow_from_root(state, Y_hat, x, f_orderx, rorderx,sweeps,norder);
	}
	else
	{
		this->right->grow_from_root(state, Y_hat, x, f_orderx, rorderx,sweeps,norder);	
		this->left->grow_from_root(state, Y_hat, x, f_orderx, lorderx,sweeps,norder);
			
	}

		
	delete lorderx;
	delete rorderx;

	return;
}


double tree_loglik(Node* node, int node_depth, double gamma, double beta) 
{
  double out = 0.0;
  if(node->is_leaf) {
    out += log(1.0 - growth_prior(node_depth, gamma, beta));
  }
  else {
    out += log(growth_prior(node_depth, gamma, beta));
    out += tree_loglik(node->left, node_depth + 1, gamma, beta);
    out += tree_loglik(node->right, node_depth + 1, gamma, beta);
  }
  return out;
}


void leaves(Node* x, std::vector<Node*>& leafs) 
{
	if (x->is_leaf) 
  {
		leafs.push_back(x);
	}
	else 
  {
		leaves(x->left, leafs);
		leaves(x->right, leafs);
	}
}


std::vector<Node*> leaves(Node* x) 
{
	std::vector<Node*> leafs(0);
	leaves(x, leafs);
	return leafs;
}

//,unsigned int sweeps ,unsigned int norder ,unsigned int treeid ,double myrank
void BART_likelihood_all(States * state,arma::vec& Y_hat,const arma::mat & x,F_OrderX * f_orderx,OrderX * orderx,const std::vector<size_t> &subset_vars, size_t &split_var, size_t &split_point,Node *tree_pointer,bool &no_split,size_t & split_count)
{

	size_t N = orderx->N;
	size_t ind=0;


	double loglike_sum=0.0;


	 


	double loglike_max = -INFINITY;
	std::vector<double> loglike;
	size_t loglike_start;
	size_t CutCount;
	if (N < state->n_cutpoints - 1 + 2 * state->n_min)
	{
		CutCount = N - 2 * state->n_min + 1;
	}
	else
	{
		CutCount = state->n_cutpoints;
	}
	
	loglike.resize(CutCount * state->p_continuous + f_orderx->X_values.size() + 1, -INFINITY);
	loglike_start = CutCount * state->p_continuous;

	if (state->p_continuous > 0)
	{
		calculate_loglikelihood_continuous(
		CutCount,
		loglike, 
		subset_vars, 
		state,
		orderx,
		Y_hat,
		x,
		loglike_max, 
		tree_pointer);
	}

	if (state->p_categorical > 0)
	{
		calculate_loglikelihood_categorical(
			loglike,
			loglike_start,
			subset_vars,
			state,
			f_orderx,
			orderx,
			Y_hat,
			x,
			loglike_max,
			tree_pointer);
	}

	calculate_likelihood_no_split(
		loglike,
		loglike_max,
		state,
		tree_pointer);
		//,sweeps,norder,treeid,myrank

	for (size_t ii = 0; ii < loglike.size(); ii++)
	{
		loglike[ii] = exp(loglike[ii] - loglike_max);
		loglike_sum+=loglike[ii];
	}

    




	std::vector<size_t> candidate_index(CutCount + 2);
	seq_gen_std2(state->n_min, N - state->n_min, CutCount, candidate_index);


	double Urand=arma::randu()*loglike_sum;
	//std::cout << Urand << " random" << std::endl;
	double acum;
	acum = 0.0;
	for (size_t i = 0; i < loglike.size(); i++)
	{   if (loglike[i] != 0)
		{
			acum += loglike[i];
			if (acum >= Urand)
			{
				ind = i;
				break;
			}
		}
	}
	// if ((sweeps==0) && (norder==0) && (treeid==3) && (tree_pointer->depth==0))
	// {
	// 	arma::vec showlist = arma::zeros<arma::vec>(loglike.size()+2);
	// 	for (size_t ii = 0; ii < loglike.size(); ii++)
	// 	{
	// 		showlist(ii)= loglike[ii]; 
	// 	}
	// 	showlist(loglike.size())=Urand;
	// 	showlist(loglike.size()+1)=ind;

	// 	std::stringstream tempfnss;
	// 	tempfnss.str("");
	// 	tempfnss <<myrank << "_showlist.csv";
	// 	showlist.save(tempfnss.str(),arma::csv_ascii);  
	// }






	if (ind == loglike.size() - 1)
	{
		no_split = true;
		split_var = 0;
		split_point = 0;
	}
	else if (ind < loglike_start)
	{
		split_var = ind / CutCount;
		split_point = candidate_index[(ind % CutCount)+1]-1;
		double TempVar=x(orderx->Xorder[split_var*N + split_point], split_var);
		while ((split_point < N - 1) && (x(orderx->Xorder[split_var*N + split_point+1], split_var) == TempVar))
		{
			split_point = split_point + 1;
		}

		split_count=split_point+1;
	}
	else
	{
		size_t start;
		ind = ind - loglike_start;

		for (size_t i = 0; i < (f_orderx->variable_ind.size() - 1); i++)
		{
			if (f_orderx->variable_ind[i] <= ind && f_orderx->variable_ind[i + 1] > ind)
			{
				split_var = i;
				break;
			}
		}
		
		
		start = f_orderx->variable_ind[split_var];
		split_point = 0;
		for (size_t i = start; i <= ind ; i++)
		{
			split_point = split_point+ orderx->Xcounts[i];
		}
		
		split_count=orderx->Xcounts[ind];
						
		if (split_point == 0)
		{
			cout<<"Something   split_point == 0       wrong,stopped"<<endl;
			exit(1);
		}
		else
		{
			split_point = split_point - 1;
		}
		split_var = split_var + state->p_continuous;
	}

	return;
}


void calculate_loglikelihood_continuous(size_t cutcount,std::vector<double> &loglike,const std::vector<size_t> &subset_vars,States * state,OrderX * orderx,arma::vec& Y_hat,const arma::mat & x,double &loglike_max,Node *tree_pointer)
{
	size_t N = orderx->N;
	std::vector<size_t> candidate_index2(cutcount + 2);
	std::vector<size_t> candidate_index3(cutcount + 2);
	seq_gen_std2(state->n_min, N - state->n_min, cutcount, candidate_index2);
	for (auto i : subset_vars)
	{

		if (i < state->p_continuous)
		{

			unsigned int * Torder = orderx->Xorder + i*orderx->N;
			double temp_suff_stat;
			temp_suff_stat = 0;

			candidate_index3[cutcount + 1] = candidate_index2[cutcount + 1];
			candidate_index3[0] = candidate_index2[0];

			for (size_t j = cutcount; j > 0; j--)
			{
				candidate_index3[j] = candidate_index2[j];
				while ((x(Torder[candidate_index3[j]-1], i) == x(Torder[candidate_index3[j]], i)) && (candidate_index3[j]<candidate_index2[j + 1]) )
				{
					candidate_index3[j] = candidate_index3[j] + 1;
				}
				if (candidate_index3[j] == candidate_index2[j + 1])
				{
					candidate_index3[j] = candidate_index3[j + 1];
				}
			}	
			
			for (size_t j = 0; j < cutcount; j++)
			{
				
				calcSuffStat_continuous(
					temp_suff_stat,
					Torder,
					candidate_index3,
					j,
					Y_hat);
				if ((candidate_index3[j + 1] != candidate_index3[j + 2]) && ( candidate_index3[j + 1] != candidate_index3[cutcount + 1] ) )
				{
					loglike[cutcount * i + j] = 
					likelihood(temp_suff_stat, tree_pointer->suff_stat, candidate_index3[j + 1], true, false, state) +
					likelihood(temp_suff_stat, tree_pointer->suff_stat, candidate_index3[j + 1], false, false, state);
					
					loglike_max = loglike_max > loglike[cutcount * i + j] ? loglike_max : loglike[cutcount * i + j];
				}
			}
		}
	}
}


void calcSuffStat_continuous(double &temp_suff_stat,unsigned int * xorder,std::vector<size_t> &candidate_index,size_t index,arma::vec& Y_hat)
{

	for (size_t q = candidate_index[index] ; q <= candidate_index[index + 1]-1; q++)
	{
		incSuffStat(Y_hat, xorder[q], temp_suff_stat);
	}
	return;
}


void incSuffStat(arma::vec& Y_hat,size_t index_next_obs, double &suffstats)
{
	suffstats += Y_hat[index_next_obs];
	return;
}


double likelihood(double & temp_suff_stat, double * suff_stat_all, size_t N_left, bool left_side, bool no_split, States *state) 
{
	double sigma2 = state->sigma2;
	size_t nb;
	double nbtau;
	double y_sum;


	if (no_split)
	{
		nb = suff_stat_all[2];
		nbtau = nb * pow(state->sigma_mu, 2);
		y_sum = suff_stat_all[0];

	}
	else if (left_side)
	{

		nb = N_left ;
		nbtau = nb * pow(state->sigma_mu, 2);
		y_sum = temp_suff_stat;
	}
	else
	{
		nb = suff_stat_all[2] - N_left ;
		nbtau = nb * pow(state->sigma_mu, 2);
		y_sum = suff_stat_all[0] - temp_suff_stat;
	}
	//6.27 ADD 0.5 * log(sigma2)
	return   0.5 * log(sigma2)  -0.5 * log(nbtau + sigma2) + 0.5 * pow(state->sigma_mu, 2) * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));
}


double likelihood(double * suff_stat_all,States *state)
{
	double sigma2 = state->sigma2;
	size_t nb;
	double nbtau;
	double y_sum;


	nb = (size_t)(suff_stat_all[2]);
	nbtau = nb * pow(state->sigma_mu, 2);
	y_sum = suff_stat_all[0];
    //6.27 ADD 0.5 * log(sigma2)
	return   0.5 * log(sigma2)  -0.5 * log(nbtau + sigma2) + 0.5 * pow(state->sigma_mu, 2) * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));
}


void calculate_loglikelihood_categorical(std::vector<double> &loglike,size_t &loglike_start,const std::vector<size_t> &subset_vars,States * state,F_OrderX * f_orderx,OrderX * orderx,arma::vec& Y_hat,const arma::mat & x,	double &loglike_max,Node *tree_pointer)
{

	for (size_t var_i = 0; var_i < subset_vars.size(); var_i++)
	{
		size_t i = subset_vars[var_i]; 
		if ((i >= state->p_continuous) && (orderx->X_num_unique[i - state->p_continuous] > 1))
		{

			double temp_suff_stat;
			size_t start, end,  n1, n2;

			start = f_orderx->variable_ind[i - state->p_continuous];
			end = f_orderx->variable_ind[i + 1 - state->p_continuous] - 1; 

			unsigned int * Torder = orderx->Xorder + i*orderx->N;
			n1 = 0;
			n2 = 0;
			for (size_t j = start; j <= end; j++)
			{
				temp_suff_stat = 0;
				n2 = n1 + orderx->Xcounts[j]-1;
				if ((orderx->Xcounts[j] >= state->n_min) && (orderx->Xcounts[j] <= orderx->N - state->n_min))
				{
					calcSuffStat_categorical(temp_suff_stat, Torder, n1, n2, Y_hat);


					loglike[loglike_start + j] = 
					likelihood(temp_suff_stat, tree_pointer->suff_stat, orderx->Xcounts[j], true, false, state) +
					likelihood(temp_suff_stat, tree_pointer->suff_stat, orderx->Xcounts[j], false, false, state);

					loglike_max = loglike_max > loglike[loglike_start + j] ? loglike_max : loglike[loglike_start + j];

				}
				n1 = n1 + orderx->Xcounts[j];

			}

		}
	}

}


void calcSuffStat_categorical(double &temp_suff_stat,unsigned int * xorder,size_t start,size_t end,arma::vec& Y_hat)
{
	for (size_t i = start; i <= end; i++)
	{
		incSuffStat(Y_hat, xorder[i], temp_suff_stat);
	}
	return;
}

//,unsigned int sweeps ,unsigned int norder ,unsigned int treeid ,double myrank
void calculate_likelihood_no_split(std::vector<double> &loglike, double &loglike_max,States * state,Node *tree_pointer)
{
	size_t loglike_size = 0;
	for (size_t i = 0; i < loglike.size(); i++)
	{
		if (loglike[i] > -INFINITY)
		{
			loglike_size += 1;
		}
	}
	if (loglike_size > 0)
	{
		loglike[loglike.size() - 1] =
		likelihood(tree_pointer->suff_stat, state)+
		log(pow(1.0 + tree_pointer->depth, state->beta) / state->gamma - 1.0)+
		log((double)loglike_size)+
		log(state->no_split_penality);

		// if ((sweeps==0) && (norder==0) && (treeid==3) && (tree_pointer->depth==0))
		// {
			
		// 	std::stringstream srnstr; 
		// 	srnstr.str("");
		// 	srnstr <<"no_split"<<myrank;
		// 	srnstr <<"P1"<<likelihood(tree_pointer->suff_stat, state)<<"\t";
		// 	srnstr <<"P2"<<log(pow(1.0 + tree_pointer->depth, state->beta) / state->gamma - 1.0)<<"\t";
		// 	srnstr <<"P3"<<log((double)loglike_size)<<"\t";
		// 	srnstr <<"P4"<<log(state->no_split_penality);
		// 	srnstr<<"\n";
		// 	cout << srnstr.str()<<std::flush;

		// }



		//attention No split penalty
	}
	else
	{
		loglike[loglike.size() - 1] = 1;
	}


	if (loglike[loglike.size() - 1] > loglike_max)
	{
		loglike_max = loglike[loglike.size() - 1];
	}
}


std::ostream &operator<<(std::ostream &os, Node *t)
{
	std::vector<Node*>  leafs;
	t->getnodes(leafs);
	os << leafs.size() << "," << leafs[0]->tau <<",," << std::endl;
	for (size_t i = 0; i <leafs.size(); i++)
	{
		os << leafs[i]->nid << ",";
		os << leafs[i]->var << ",";
		os << leafs[i]->val<< ",";
		os << leafs[i]->mu<< std::endl;
	}
	return os;
}


// double LogLT(Node* n, const arma::vec& Y, const arma::mat& X, States *state)
// {
// 	std::vector<Node*> leafs = leaves(n);
// 	int num_leaves = leafs.size();

// 	arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
// 	arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
// 	arma::vec w_i = arma::zeros<arma::vec>(num_leaves);
// 	arma::mat Lambda = arma::zeros<arma::mat>(num_leaves, num_leaves);

// 	for (unsigned int i = 0; i < X.n_rows; i++) {
// 		n->GetW(X, i, state);
// 		for (int j = 0; j < num_leaves; j++) {
// 			w_i(j) = leafs[j]->current_weight;
// 		}
// 		mu_hat = mu_hat + Y(i) * w_i;
// 		Lambda = Lambda + w_i * trans(w_i);
// 	}

// 	Lambda = Lambda / state->sigma2;
// 	mu_hat = mu_hat / state->sigma2;
// 	Omega_inv = Lambda + arma::eye(num_leaves, num_leaves) / pow(state->sigma_mu, 2);
// 	mu_hat = solve(Omega_inv, mu_hat);
// 	double out = -0.5*num_leaves * (log(M_2_PI)+2*log(state->sigma_mu));
// 	double val, sign;
// 	log_det(val, sign, Omega_inv / M_2_PI);
// 	out -= 0.5 * val;
// 	out -= 0.5 * dot(Y, Y) / state->sigma2;
// 	out += 0.5 * dot(mu_hat, Omega_inv * mu_hat);
// 	return out;
// }


double LogLT(Node* n, const arma::vec& Y, const arma::mat& X, States *state)
{
	std::vector<Node*> leafs = leaves(n);
	int num_leaves = leafs.size();

	arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
	arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
	arma::vec w_i = arma::zeros<arma::vec>(num_leaves);
	arma::mat Lambda = arma::zeros<arma::mat>(num_leaves, num_leaves);

	for (unsigned int i = 0; i < X.n_rows; i++) {
		n->GetW(X, i, state);
		for (int j = 0; j < num_leaves; j++) {
			w_i(j) = leafs[j]->current_weight;
		}
		mu_hat = mu_hat + Y(i) * w_i;
		Lambda = Lambda + w_i * trans(w_i);
	}

	Lambda = Lambda / state->sigma2;
	mu_hat = mu_hat / state->sigma2;
	Omega_inv = Lambda + arma::eye(num_leaves, num_leaves) / pow(state->sigma_mu, 2);
//  mu_hat = solve(Omega_inv, mu_hat);
	double out = -1.0*num_leaves * log(state->sigma_mu);
	double val, sign;
	log_det(val, sign, Omega_inv);
	out -= 0.5 * val;
//	out -= 0.5 * dot(Y, Y) / state->sigma2;
	out += 0.5 * dot(mu_hat, inv_sympd(Omega_inv)* mu_hat);
	return out;
}

double LogLTB(Node* n, const arma::vec& Y, const arma::mat& X, States *state,int Mucount,double * Mulist)
{
    std::vector<Node*> leafs = leaves(n);
    int num_leaves=Mucount;

	arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
	arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
	arma::vec w_i = arma::zeros<arma::vec>(num_leaves);
	arma::mat Lambda = arma::zeros<arma::mat>(num_leaves, num_leaves);

	for (unsigned int i = 0; i < X.n_rows; i++) {
		n->GetW(X, i, state);
		for (int j = 0; j < num_leaves; j++) {
			w_i(j) = leafs[j]->current_weight;
		}
		mu_hat = mu_hat + Y(i) * w_i;
		Lambda = Lambda + w_i * trans(w_i);
	}

	Lambda = Lambda / state->sigma2;
	mu_hat = mu_hat / state->sigma2;
	Omega_inv = Lambda + arma::eye(num_leaves, num_leaves) / pow(state->sigma_mu, 2);
    mu_hat = solve(Omega_inv, mu_hat);

    arma::vec mu_samp = rmvnorm(mu_hat, Omega_inv);
    for(int i=0; i<Mucount; i++) 
    {
      Mulist[i]=mu_samp[i];
    }  


	double out = -1.0*num_leaves * log(state->sigma_mu);
	double val, sign;
	log_det(val, sign, Omega_inv);
	out -= 0.5 * val;
//	out -= 0.5 * dot(Y, Y) / state->sigma2;
	out += 0.5 * dot(mu_hat, Omega_inv* mu_hat);
	return out;
}

double LogLTA(Node* n, const arma::vec& Y,const arma::mat& X, States *state,const double myrank,int Mucount,double * Mulist ) 
{
  double out;
  int num_leaves=Mucount;
  
  arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
  arma::mat Omega_inv = arma::zeros<arma::mat>(num_leaves, num_leaves);
  GetSuffStats(n, Y, X, state, mu_hat, Omega_inv,myrank);

  if (myrank==0)
  { 
    arma::vec mu_samp = rmvnorm(mu_hat, Omega_inv);
    for(int i=0; i<Mucount; i++) 
    {
      Mulist[i]=mu_samp[i];
    }  

    out =  -0.5*num_leaves * (log(M_2_PI)+2*log(state->sigma_mu));
    double val, sign;
    log_det(val, sign, Omega_inv / M_2_PI);
    out -= 0.5 * val;
    out += 0.5 * dot(mu_hat, Omega_inv * mu_hat);
  }
  else
  {
    out=0;
  }
  return out;
}


double Node::updateWidth(const arma::vec& Y,const arma::mat& X, States * state)
{
	bool bupdateWidth = false;
	std::vector<Node *>  Nv;
	getnodes(Nv);
	for (size_t i = 0; i<Nv.size(); i++)
	{
		if  (!(Nv[i]->is_leaf))
		{
			size_t Temp_v = Nv[i]->var;
			if (Temp_v <state->p_continuous)
			{
				bupdateWidth = true;
				break;
			}
		}
	}
	double Maxresult = -INFINITY;
	if (bupdateWidth)
	{
		double tau_old = r_exp(state->tau_rate);
		SetTau(tau_old);
		double l_old = LogLT(this, Y, X, state);
		double loglik_old = l_old + logprior_tau(tau_old, state->tau_rate);
		double new_to_old = log_tau_trans(tau_old);
		for (size_t i = 0; i < state->RepeatT; i++)		
		{	double tau_new = tau_proposal(tau);
			SetTau(tau_new);
			double l_new = LogLT(this, Y, X, state);
			double loglik_new = l_new + logprior_tau(tau_new, state->tau_rate);
			double old_to_new = log_tau_trans(tau_new);
			bool accept_mh = do_mh(loglik_new, loglik_old, new_to_old, old_to_new);
			//std::cout << "OLD " << l_old << "New " << l_new << "Choose" << (accept_mh ? tau_new : tau_old) << std::endl;
			if (accept_mh)
			{
				tau_old = tau_new;
				l_old = l_new;
				loglik_old = loglik_new;
				new_to_old = old_to_new;
				Maxresult=l_new;
			}
			else
			{
				SetTau(tau_old);
				Maxresult=l_old;
			}
		}		
	}
	else
	{
		Maxresult = LogLT(this,Y,X,state);
	}
	return Maxresult;
}


double Node::updateWidth_Opt(const arma::vec& Y,const arma::mat& X, States * state)
{
	//std::stringstream srnstr; 
	bool bupdateWidth = false;
	std::vector<Node *>  Nv;
	getnodes(Nv);
	for (size_t i = 0; i<Nv.size(); i++)
	{
		if  (!(Nv[i]->is_leaf))
		{
			size_t Temp_v = Nv[i]->var;
			if (Temp_v <state->p_continuous)
			{
				bupdateWidth = true;
				break;
			}
		}
	}
	double Maxresult = -INFINITY;
	double Maxtau=0;

	if (bupdateWidth)
	{
		
		arma::vec StepS = arma::linspace(0, StepSize, state->RepeatT);
		arma::vec StepR = arma::randu<arma::vec>(state->RepeatT)*StepSize/ (state->RepeatT-1.0)  ;
		StepR(state->RepeatT-1)=arma::randu()*StepSize;
		StepS =StepR+StepS;

	    // srnstr.str("STEP");
	 	// for(unsigned int i=0;i<state->RepeatT;i++)
		// {
		//     srnstr<<" "<<i <<" " <<StepS[i];    
		// }
		// srnstr<< " \n"<<std::ends;
		// std::cout << srnstr.str()<<std::flush;

		for(unsigned int i=0;i<state->RepeatT;i++)
		{
			double tau_old = StepS(i);
			SetTau(tau_old);
			double l_old = LogLT(this, Y, X, state);
			if (Maxresult<l_old)
			{
				Maxresult =l_old;
				Maxtau=tau_old;

			}
            
		}
        SetTau(Maxtau);   

	}
	else
	{
		Maxresult = LogLT(this,Y,X,state);
	}
	return Maxresult;
}


arma::vec predict(Node* n, const arma::mat& X, States * state) 
{
	std::vector<Node*> leafs = leaves(n);
	int num_leaves = leafs.size();
	int N = X.n_rows;
	arma::vec out = arma::zeros<arma::vec>(N);
	for (int i = 0; i < N; i++) 
	{
		n->GetW(X, i,state);
		for (int j = 0; j < num_leaves; j++) {
			out(i) = out(i) + leafs[j]->current_weight * leafs[j]->mu;
		}
	}
	return out;
}


std::vector<Node*> init_forest(States * state) 
{
	std::vector<Node*> forest(0);
	for (unsigned int t = 0; t < state->num_trees; t++) {
		Node* n = new Node;
		n->Root();
		forest.push_back(n);
	}
	return forest;
}


void GetSuffStats(Node* n, const arma::vec& y,const arma::mat& X, States *  state,arma::vec& mu_hat_out, arma::mat& Omega_inv_out) 
{
	std::vector<Node*> leafs = leaves(n);
	int num_leaves = leafs.size();
	arma::vec w_i = arma::zeros<arma::vec>(num_leaves);
	arma::vec mu_hat = arma::zeros<arma::vec>(num_leaves);
	arma::mat Lambda = arma::zeros<arma::mat>(num_leaves, num_leaves);

	for (unsigned int i = 0; i < X.n_rows; i++) {
		n->GetW(X, i,state);
		for (int j = 0; j < num_leaves; j++) {
			w_i(j) = leafs[j]->current_weight;
		}
		mu_hat = mu_hat + y(i) * w_i;
		Lambda = Lambda + w_i * trans(w_i);
	}
	Lambda = Lambda / state->sigma2 ;
	mu_hat = mu_hat / state->sigma2;
	Omega_inv_out = Lambda + arma::eye(num_leaves, num_leaves) / pow(state->sigma_mu, 2);
	mu_hat_out = solve(Omega_inv_out, mu_hat);
}


void GetSuffStats(Node* n, const arma::vec& y,const arma::mat& X, States *  state,arma::vec& mu_hat_out, arma::mat& Omega_inv_out,const double myrank) 
{
  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  arma::vec w_i = arma::zeros<arma::vec>(num_leaves);
  arma::vec mu_hat_S = arma::zeros<arma::vec>(num_leaves);
  arma::mat Lambda_S = arma::zeros<arma::mat>(num_leaves, num_leaves);
  arma::vec mu_hat_H = arma::zeros<arma::vec>(num_leaves);
  arma::mat Lambda_H = arma::zeros<arma::mat>(num_leaves, num_leaves);

  if (num_leaves==1)
  {
    mu_hat_S[0]=arma::sum(y);
    Lambda_S(0,0)=X.n_rows;
  }
  else
  {
    for(unsigned int i = 0; i < X.n_rows; i++) 
	{
      n->GetW(X, i,state);
      for(int j = 0; j < num_leaves; j++) 
	  {
        w_i(j) = leafs[j]->current_weight;
      }
      mu_hat_S = mu_hat_S + y(i) * w_i;
      Lambda_S = Lambda_S + w_i * arma::trans(w_i);
    }
  }

  double* mu_hat_S_mem = mu_hat_S.memptr();
  double* Lambda_S_mem = Lambda_S.memptr();

  double* mu_hat_H_mem = mu_hat_H.memptr();
  double* Lambda_H_mem = Lambda_H.memptr();


  MPI_Reduce(mu_hat_S_mem, mu_hat_H_mem, num_leaves, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(Lambda_S_mem, Lambda_H_mem, num_leaves*num_leaves, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myrank==0)
  {
    Lambda_H = Lambda_H / pow(state->sigma, 2) * state->temperature;
    mu_hat_H = mu_hat_H / pow(state->sigma, 2) * state->temperature;

    Omega_inv_out = Lambda_H + arma::eye(num_leaves, num_leaves) / pow(state->sigma_mu, 2);
    mu_hat_out = solve(Omega_inv_out, mu_hat_H);

  }  

}


double ldcauchy(double x, double loc, double sig)
{
	return log(sig / PI / (sig*sig + (x - loc)*(x - loc)));
}


double cauchy_jacobian(double tau, double sigma_hat)
{
	double sigma = pow(tau, -0.5);
	double out = ldcauchy(sigma, 0.0, sigma_hat);
	out = out - M_LN2 - 3.0 / 2.0 * log(tau);
	return out;
}










int main(int argc, char** argv)
{ 
	std::stringstream srnstr; 
	std::stringstream tempfnss;  
	std::vector<Node*> forest; 
	int myrank, processes;

	arma::mat X;  
	arma::mat X_TEST; 
	arma::vec Y;
	arma::vec RY;
	arma::vec RES;


	//clock_t RTime;

	double n_local;
	double n_global;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	MPI_Comm_size(MPI_COMM_WORLD,&processes);

	srnstr<< "MPI: node " << myrank << " of " << processes << " processes.\n"<<std::ends;
	cout << srnstr.str()<<std::flush;

	if(argc!=48) {cout << argc<<"Parameter not match\n"; return 1;}

	arma::arma_rng::set_seed(myrank+888); 

	States * states=new States();
	states->alpha=atof(argv[2]);
	states->beta=atof(argv[3]);
	states->gamma=atof(argv[4]);
	states->sigma_hat=atof(argv[6]);
	states->shape=atof(argv[7]);
	states->no_split_penality=atof(argv[8]);
	states->num_trees=atoi(argv[9]);
	states->alpha_scale=atof(argv[10]);
	states->alpha_shape_1=atof(argv[11]);
	states->alpha_shape_2=atof(argv[12]);
	states->tau_rate=atof(argv[13]);
	states->num_burn=atoi(argv[14]);
	states->num_save=atoi(argv[15]);
	states->update_sigma_mu=atoi(argv[16]);
	states->update_s=atoi(argv[17]);
	states->update_alpha=atoi(argv[18]);
	states->update_beta=atoi(argv[19]);
	states->update_gamma=atoi(argv[20]);
	states->update_tau=atoi(argv[21]);
	states->update_tau_mean=atoi(argv[22]);
	states->verbose=atoi(argv[23]);
	states->binary=atoi(argv[24]);
	states->binaryOffset=atof(argv[25]);
	states->n_min=atoi(argv[26]);
	states->p_categorical=atoi(argv[27]);
	states->p_continuous=atoi(argv[28]);
	states->max_depth=atoi(argv[29]);
	states->n_cutpoints=atoi(argv[30]);
	states->RepeatT=atoi(argv[31]);
	states->YMax=atof(argv[32]);
	states->YMin=atof(argv[33]);
	states->sigma_mu=atof(argv[34]);
	states->mtry=atoi(argv[35]);                                        //unused 
	states->Selected=atoi(argv[36]);                                        //unused  domh->Selected 
	states->depthconstrain=atoi(argv[37]);                              //REused
	states->delhigest=atoi(argv[38]);                                   //unused
	states->widetype=atoi(argv[39]);
	states->Try=atoi(argv[40]);
	states->mixrestart=atoi(argv[41]);                                  //unused
	states->selectType=atoi(argv[42]);
	states->winnercontrol=atoi(argv[43]);                               //unused
	states->TreeCount[0]=0;
	states->TreeCount[1]=states->num_trees;
	states->TreeCount[2]=states->TreeCount[1] + atoi(argv[45]);
	states->TreeCount[3]=states->TreeCount[2]+atoi(argv[46]);
	states->Horder=atoi(argv[44]);
	states->num_trees=states->TreeCount[3];
	states->resetP=atof(argv[47]);
	
	tempfnss.str("");
	tempfnss << argv[1] << "x"<<myrank<<".csv";
	X.load(tempfnss.str(),arma::csv_ascii);
	
	tempfnss.str("");
	tempfnss << argv[1] << "xp"<<myrank<<".csv";
	X_TEST.load(tempfnss.str(),arma::csv_ascii);

	states->sigma=states->sigma_hat;
	states->temperature=1.0;
	states->sigma_mu_hat=states->sigma_mu;
	states->sigma2=states->sigma*states->sigma;
	states->np=processes-1;
	states->p=states->p_continuous+states->p_categorical;  
	states->n=X.n_rows;
	states->n_test=X_TEST.n_rows;


	states->num_sweeps=states->num_save*states->Try+states->num_burn;


	states->InitStates();

	if (myrank==0 && states->verbose)
	{
		states->ShowContents();
	}

	n_local=X.n_rows;

	tempfnss.str("");
	tempfnss << argv[1] << "y"<<myrank<<".csv";
	RY.load(tempfnss.str(),arma::csv_ascii);

	Y = arma::zeros<arma::vec>(X.n_rows);
	Y=RY*1.0;


	n_global=0; 
	MPI_Reduce(&n_local, &n_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (myrank==0)
	{
		states->totalcount=n_global;
		if (states->verbose)
		{
			srnstr.str("");
			srnstr<<"Total Training Size "<<states->totalcount<<"\nLocal Test Size"<<X_TEST.n_rows << "\n";
			cout << srnstr.str()<<std::flush;
		}
	}

	F_OrderX * f_orderx = new F_OrderX(states->p_categorical + 1);
	OrderX * orderx = new  OrderX(X, f_orderx, states);

	forest = init_forest(states);
	if (states->Selected)
	{
		tempfnss.str("");
		tempfnss << "Selected.csv";
		states->Relation.load(tempfnss.str(),arma::csv_ascii);

		states->Pages=arma::zeros<arma::mat>(states->Relation.n_rows ,states->p+3);

		unsigned int noworder=1;
		for (unsigned int i = 0; i < states->Relation.n_rows; i++) 
		{
			unsigned int Relationi=states->Relation(i,0);
			states->Level[Relationi]=states->Level[Relationi]+1;
			if (noworder>Relationi)
			{
				srnstr.str("");
				srnstr<<"wrong relation order,please check it\n";
				cout << srnstr.str()<<std::flush;
				exit(1);
			}
			noworder=Relationi;
			for (unsigned int j = 1; j <=2; j++) 
		    {
				if ((states->Relation(i,j)<0) ||  (states->Relation(i,j)>=states->p))
				{
					srnstr.str("");
				    srnstr<<"wrong relation table,please check it\n";
				    cout << srnstr.str()<<std::flush;
					exit(1);
				}
			}

			if 	((Relationi==2) && (states->Relation(i,1)>=states->Relation(i,2)))
			{
				srnstr.str("");
				srnstr<<"wrong relation table,please check it\n";
				cout << srnstr.str()<<std::flush;
				exit(1);
			}

			if (Relationi==1)
			{
				states->Pages(i,states->Relation(i,1))=1.0;
				states->Pages(i,states->p)=states->Relation(i,1);

				if (i==0)
				{
					states->Pages(i,states->p+1)=TCut;
				}
				else
				{
					states->Pages(i,states->p+1)=states->Pages(i-1,states->p+1)+TCut;
				}
				
			}
			else if(Relationi==2)
			{
				states->Pages(i,states->Relation(i,1))=1.0;
				states->Pages(i,states->Relation(i,2))=1.0;

				states->Pages(i,states->p)=states->Relation(i,1)*states->p + states->Relation(i,2) + states->p - (states->Relation(i,1)+1)*(states->Relation(i,1)+2)/2.0 ;
				if (i==0)
				{
					states->Pages(i,states->p+1)=TCut*TCut;
				}
				else
				{
					states->Pages(i,states->p+1)=states->Pages(i-1,states->p+1)+TCut*TCut;
				}

			}
			states->Pages(i,states->p+2)=Relationi;




			if ((states->Relation(i,0)==1) && (states->Relation(i,2) != 0 ))
			{
				srnstr.str("");
				srnstr<<"wrong relation table,please check it\n";
				cout << srnstr.str()<<std::flush;
				exit(1);
			}   
		}
		states->Level[2]=states->Level[2]+states->Level[1];
		states->PartialX=arma::zeros<arma::mat>(states->Pages(states->Relation.n_rows-1,states->p+1),states->p+1);

		for (unsigned int i = 0; i < states->Relation.n_rows; i++) 
		{
			int Relationi=states->Relation(i,0);
			int iStart;
			unsigned int ival1=states->Relation(i,1);
			unsigned int ival2=states->Relation(i,2);
			if (i==0)
			{
				iStart=0;
			}
			else
			{
				iStart=states->Pages(i-1,states->p+1);
			}
			if (Relationi==1)
			{
				for (unsigned int j1 = 0; j1 <TCut; j1++) 
		   		{
					states->PartialX(iStart+j1,ival1)=j1*1.0/(TCut-1.0);
				}			
			}
			else
			{
				for (unsigned int j1 = 0; j1 <TCut*TCut; j1++) 
		   		{
					states->PartialX(iStart+j1,ival1)=1.0/(TCut-1.0)*(j1 % TCut )   ;
					states->PartialX(iStart+j1,ival2)=1.0/(TCut-1.0)*(j1 / TCut )   ;
				}	

			} 

		}	


	}



    arma::vec Remember_to_del = arma::zeros<arma::vec>(states->num_trees);
	arma::vec Winner_List =	arma::zeros<arma::vec>(states->num_trees) * (states->np +1) ;
	arma::mat y_train_ave = arma::zeros<arma::mat>(X.n_rows, states->Horder); 
	arma::mat y_test_ave = arma::zeros<arma::mat>(states->n_test, states->Horder); 

	for (unsigned int kk = 0; kk < states->Horder; kk++) 
	{	
        states->y_hat_train = arma::zeros<arma::mat>(states->num_save, states->n);
		states->y_hat_test = arma::zeros<arma::mat>(states->num_save, states->n_test);
		
		RES = Y;

		if (kk>0)
		{
			states->pmin= pow(2.0 * states->n_min/n_local,1.0/kk);
		}  

		for (unsigned int i = 0; i < states->num_sweeps; i++) 
		{
			std::vector<double> means(0);

			states->split_count_all_tree.zeros();
			
			if (states->depthconstrain)
			{
				int iDepth;
				if (i<states->num_burn)
				{
					iDepth=i+1+kk;
				}
				else
				{
					iDepth=((i - states->num_burn) % states->Try)+1+kk;
				}

				if (iDepth >states->MD )
				{
					states->max_depth=states->MD;
				}
				else
				{
					states->max_depth=iDepth;
				}
			}


			for (unsigned int j = states->TreeCount[kk]; j < states->TreeCount[kk+1]; j++)
			{
				RES = RES + states->y_hat_all.col(j);

				Node * myNode = new Node();
				myNode->Root();
				initialize_root_suffstat(RES, myNode->suff_stat);
			    //,j,myrank
				myNode->grow_from_root(states,RES,X,f_orderx,orderx,i,kk);

				if (myNode->height()<=kk)
				{
					double Temp_S[3];
					Temp_S[0]=myNode->suff_stat[0];
					Temp_S[1]=myNode->suff_stat[1];
					Temp_S[2]=myNode->suff_stat[2];
					
					delete myNode; 
					myNode = new Node();
					myNode->Root();
					
					myNode->suff_stat[0]=Temp_S[0];
					myNode->suff_stat[1]=Temp_S[1];
					myNode->suff_stat[2]=Temp_S[2];
				}

				double loglik;

				if (states->widetype==1)
				{
					if (states->selectType==2)
					{
						loglik=myNode->updateWidth_Opt(RES, X, states);
						loglik=1;
					}else 
					{
						loglik=myNode->updateWidth_Opt(RES, X, states);
						if (states->selectType != 4 )
						{
							double TLoglik=	tree_loglik(myNode,0,states->gamma,states->beta);
							loglik=loglik+TLoglik;
						}
					}
				}
				else
				{

					if (states->selectType==2)
					{
						loglik=myNode->updateWidth(RES, X, states);
						loglik=1;
					}else 
					{
						loglik=myNode->updateWidth(RES, X, states);
						if (states->selectType != 4 )
						{
							double TLoglik=	tree_loglik(myNode,0,states->gamma,states->beta);
							loglik=loglik+TLoglik;
						}
					}
				}
				
				int Winner=Collect_Sample(loglik,myrank,states);

				double * TreePara=new double[2];
				std::vector<Node*>  leafs;

				if(Winner==((int)myrank))
				{
					myNode->getnodes(leafs);
					TreePara[0]=leafs.size();
					TreePara[1]=myNode->tau;
				}

				MPI_Bcast(TreePara,2,MPI_DOUBLE,Winner,MPI_COMM_WORLD);

				int Nodesize=(int) TreePara[0];
				double tau_t =TreePara[1];

				delete [] TreePara;

				double * Tree_Memory=new double[3*Nodesize];

				if(Winner==((int)myrank))
				{

					for (int t1 = 0; t1 < Nodesize; t1++)
					{
						Tree_Memory[t1*3]=leafs[t1]->nid;
						Tree_Memory[t1*3+1]=leafs[t1]->val;
						Tree_Memory[t1*3+2]=leafs[t1]->var;						
					}
				}

				MPI_Bcast(Tree_Memory,3*Nodesize,MPI_DOUBLE,Winner,MPI_COMM_WORLD);

				if(Winner != ((int)myrank))
				{
					delete myNode; 
					myNode = new Node();
					myNode->Root();
					myNode->val=Tree_Memory[1];
					myNode->var=Tree_Memory[2];
					myNode->tau=tau_t;

					for (int t1 = 1; t1 < Nodesize; t1++)
					{
						Node * TNode = new Node();
						TNode->nid=Tree_Memory[3*t1];
						TNode->val=Tree_Memory[3*t1+1];
						TNode->var=Tree_Memory[3*t1+2];
						TNode->tau=tau_t;
						Node * PNode=myNode->getptr(TNode->nid/2);
						PNode->is_leaf=false;
						TNode->depth=PNode->depth+1;
						TNode->parent=PNode;
						if(2* PNode->nid == TNode->nid )
						{
							PNode->left=TNode;
						}
						else
						{
							PNode->right=TNode;
						}
					}
				}

				delete [] Tree_Memory;

				int nbot_B = myNode->nbots();
				double * MUlist_B =new double [nbot_B];
				int Change=0;
				int nbot_A = forest[j]->nbots();
				double * MUlist_A =new double [nbot_A];

				
				if (myrank==0)
				{
					double loglik_B1 = LogLTB(myNode, RES, X, states,nbot_B,MUlist_B);
					double loglik_B2 = tree_loglik(myNode, 0, states->gamma, states->beta);
					double loglik_B =loglik_B1+loglik_B2;

					double loglik_A1 = LogLTB(forest[j], RES, X, states,nbot_A,MUlist_A) ;
					double loglik_A2 = tree_loglik(forest[j], 0, states->gamma, states->beta);
					double loglik_A=loglik_A1+loglik_A2;

					if (do_mh(loglik_B, loglik_A, 0, 0))
					{
						Change=1;
					}
					else if (( i>=states->num_burn )   &&  ((i - states->num_burn) % states->Try ==0) && (arma::randu()<=states->resetP))
					{
						Change=2;
					}

				}
				MPI_Bcast(&Change,1,MPI_INT,0,MPI_COMM_WORLD);

				if (Change)
				{
					delete forest[j];
					forest[j] = myNode;
					MPI_Bcast(MUlist_B,nbot_B,MPI_DOUBLE,0,MPI_COMM_WORLD);
					forest[j]->UpdateMuA(MUlist_B,nbot_B,means);
					states->Node_counts(i,j)=nbot_B;
					Winner_List[j]=Winner;
				}
				else
				{
					delete myNode;
					MPI_Bcast(MUlist_A,nbot_A,MPI_DOUBLE,0,MPI_COMM_WORLD);
					forest[j]->UpdateMuA(MUlist_A,nbot_A,means);
					states->Node_counts(i,j)=nbot_A;
				}

				delete [] MUlist_A;
				states->change_histroy(i,j)=Change;

				states->tau_histroy(i,j)=forest[j]->tau;
				delete [] MUlist_B;
			
				states->split_count_current_tree.zeros();   
				forest[j]->updateSplit(states);
                int comlocate;
			
				comlocate=states->Findlocate(kk);

				if (myrank==0)
				{								
					states->split_Type(i,j)=	comlocate+1;
					if (comlocate>=0)
					{


						if  (  ( i>=states->num_burn  -1)  &&  ((i - states->num_burn +1  ) % states->Try ==0) )
						{
							states->split_count_all_tree(comlocate)=1;

							
						}







						// srnstr.str("");
						// srnstr <<"KK"<<kk<<"Wave"<<i<<"Tree"<<j<<"Type"<<comlocate; 
						// srnstr<<"\n";
						// cout << srnstr.str()<<std::flush;		

					}
					else
					{
						srnstr.str("");
						srnstr <<"Wave"<<i<<"Tree"<<j<<"KK"<<kk<<"fail"; 

						for (unsigned int t2=0;t2<states->p;t2++)
						{ 
							srnstr <<"\t"<<states->split_count_current_tree[t2];
						}
						srnstr<<"\n"; 
						cout << srnstr.str()<<std::flush;	
					}
				}
				states->y_hat_all.col(j)=predict(forest[j], X, states);





				RES = RES - states->y_hat_all.col(j);
				states->UpdateSigmaA(RES,myrank);
				if (i>=states->num_burn && myrank==0)
				{
					states->S_sigma(i-states->num_burn,j)=states->sigma;
				}

				if  (  ( i>=states->num_burn + states->Try -1)   &&  (states->n_test>0)  &&  ((i - states->num_burn- states->Try +1  ) % states->Try ==0) )
				{
					states->y_hat_test.row((i-states->num_burn - states->Try +1)/states->Try)  =  states->y_hat_test.row( (i-states->num_burn - states->Try +1)/states->Try  ) + arma::trans(predict(forest[j], X_TEST, states));
				}	


				if (myrank==0)
				{	
					if  (  ( i>=states->num_burn + states->Try -1) && ( comlocate>=0 )  &&  (states->Selected)  &&  ((i - states->num_burn- states->Try +1  ) % states->Try ==0) )
					{
						bool BFind=true;
						unsigned int t2;	
						for (t2=0;t2<states->Pages.n_rows;t2++)
						{ 
							if (states->Pages(t2,states->p)== comlocate)
							{
								BFind=false;
								break;
							}
						}
						if (BFind)
						{

							srnstr.str("");
							srnstr <<"Can not find  Type"<<comlocate; 
							srnstr << "\n";
							cout << srnstr.str()<<std::flush;	
							exit(2);
						}

						unsigned int iStart=(t2==0?0:states->Pages(t2-1,states->p+1));
                        unsigned int iLen=(states->Pages(t2,states->p+2)==1?TCut:TCut*TCut);
						arma::mat TempX=states->PartialX.submat(iStart,0,iStart+iLen-1,states->p-1);
						states->PartialX.submat(iStart,states->p,iStart+iLen-1,states->p) =states->PartialX.submat(iStart,states->p,iStart+iLen-1,states->p)  +       predict(forest[j], TempX, states);
					}
				}					
			}


			if (myrank==0)
			{								
				if  (  ( i>=states->num_burn  -1)  &&  ((i - states->num_burn +1  ) % states->Try ==0) )
				{
					states->split_count_all_tree_sum=states->split_count_all_tree_sum+states->split_count_all_tree;
				}
			}


			

			if (states->update_sigma_mu)
			{
				states->UpdateSigmaMu(means,myrank);
				if (i>=states->num_burn && myrank==0)
				{
					states->S_sigma_mu(i-states->num_burn,kk)=states->sigma_mu;
				}
			}

			if  ( ( i >= states->num_burn+ states->Try -1 )   &&        ( (i - states->num_burn- states->Try +1  ) % states->Try ==0) )
			{
				states->y_hat_train.row((i-states->num_burn - states->Try +1)/states->Try) = arma::trans((Y-RES));
			}





		}

		y_train_ave.col(kk)=arma::trans(arma::mean(states->y_hat_train,0));
		if (states->n_test>0)
		{
			y_test_ave.col(kk)=arma::trans(arma::mean(states->y_hat_test,0));
		}	
		Y=Y-y_train_ave.col(kk);
	}

	if (states->n_test>0)
	{
		tempfnss.str("");
		tempfnss << argv[1] << "_R_ytest"<<myrank<<".csv";

        arma::mat y_test_sum =  (states->YMax-states->YMin)*(arma::sum(y_test_ave,1) +0.5)+states->YMin;   

		y_test_sum.save(tempfnss.str(),arma::csv_ascii);
	}

	tempfnss.str("");
	tempfnss << argv[1] << "_R_ytrain"<<myrank<<".csv";
	y_train_ave.save(tempfnss.str(),arma::csv_ascii);

	if (myrank==0)
	{

		states->S_sigma=states->S_sigma*(states->YMax-states->YMin);
		tempfnss.str("");
		tempfnss << argv[1] << "_R_sigma"<<".csv";
		states->S_sigma.save(tempfnss.str(),arma::csv_ascii);  

		states->S_sigma_mu=states->S_sigma_mu*(states->YMax-states->YMin);
		tempfnss.str("");
		tempfnss << argv[1] << "_R_sigma_mu"<<".csv";
		states->S_sigma_mu.save(tempfnss.str(),arma::csv_ascii);  

		tempfnss.str("");
		tempfnss << argv[1] << "_R_VC"<<".csv";
		states->Node_counts.save(tempfnss.str(),arma::csv_ascii);  

		tempfnss.str("");
		tempfnss << argv[1] << "_R_CH"<<".csv";
		states->change_histroy.save(tempfnss.str(),arma::csv_ascii);  

		tempfnss.str("");
		tempfnss << argv[1] << "_R_TH"<<".csv";
		states->tau_histroy.save(tempfnss.str(),arma::csv_ascii);  

		tempfnss.str("");
		tempfnss <<argv[1] << "_split_count_all_tree"<<".csv";
		states->split_count_all_tree_sum.save(tempfnss.str(),arma::csv_ascii);  

		tempfnss.str("");
		tempfnss <<argv[1] << "_split_Type"<<".csv";
		states->split_Type.save(tempfnss.str(),arma::csv_ascii);  

		if (states->Selected)
		{

			states->PartialX.tail_cols(1)=states->PartialX.tail_cols(1)*(states->YMax-states->YMin)/states->num_save;	
			tempfnss.str("");
			tempfnss <<"PartialPlot"<<".csv";
			states->PartialX.save(tempfnss.str(),arma::csv_ascii);  
		}





		tempfnss.str("");
		tempfnss << argv[1] << "_R_TREE"<<".csv";
		std::ofstream fout(tempfnss.str());

		for (unsigned int j = 0; j < states->num_trees; j++)
		{
			fout<<forest[j];
		}
		fout.close();
	}

	delete orderx;
	MPI_Finalize();
	return 0;
}	

