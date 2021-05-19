//=====================================================
//Copyright (c) 2015. Yu-Cheng Ku. All Rights Reserved.
//=====================================================

#include <oxstd.h>
#include "mcmc_sv.h"       // header for sv.dll
#include "arms.h"	       // header for arms.dll
#include "../ssfpack/ssfpack.h"  // header for ssfpack.dll
#include "oxprob.h"

decl d_post;
decl k_post;
static decl d_true = 0.8;
static decl k_true = 10;
static decl T      = 500;

// Class for posterior of d	======================================================
class postd
{
	decl psi, Ainv, Qs, k, T, q;
	post_d(const d);
	set_other(const new_psi, const new_Ainv, const new_Qs, const new_k, const new_T, const new_q);
};

postd::post_d(const d)
{
	decl Cinv_d = zeros(q,q);
	decl prod_d = new array[T];
		 prod_d[0:T-1] = zeros(q,q);
	  
	decl Q_sqrt, Qu, Qd, Qv;
	for(decl t=0; t<T; ++t)
	{
		decsvd(Qs[t][][], &Qu, &Qd, &Qv);
		Q_sqrt = Qu * diag(Qd.^(d/2)) * Qv';
 
		prod_d[t][][] = Q_sqrt * invert(Qs[t+1][][]) * Q_sqrt;
		Cinv_d += prod_d[t][][];
	}
	 
	return psi*d - 0.5*k*trace(Ainv*Cinv_d);
}

postd::set_other(const new_psi, const new_Ainv, const new_Qs, const new_k, const new_T, const new_q)
{
	psi  = new_psi;
	Ainv = new_Ainv;
	Qs   = new_Qs;
	k    = new_k;
	T    = new_T;
	q    = new_q;
}

post_d(const d)
{
	return d_post -> post_d(d);
}
//================================================================================

// Class for posterior of k ======================================================
class postk
{
	decl lambda0, A, Cinv_k, sum_log_Q, T, q;
	post_k(const k);
	set_other_k(const new_lambda0, const new_A, const new_Cinv_k, const new_sum_log_Q, const new_T, const new_q); 
}

postk::post_k(const k)
{
	return -lambda0*k + 0.5*T*k*q*log(k/2) - 0.5*T*k*log(determinant(A))
		   - T*sumr(log(gammafact((k+1-range(1,q))/2))) + (k/2)*sum_log_Q - 0.5*k*trace(invert(A)*Cinv_k); 
}

postk::set_other_k(const new_lambda0, const new_A, const new_Cinv_k, const new_sum_log_Q, const new_T, const new_q)
{
	lambda0   = new_lambda0;
	A         = new_A;
	Cinv_k    = new_Cinv_k;
	sum_log_Q = new_sum_log_Q;
	T         = new_T;
	q         = new_q;
}

post_k(const k)
{
	return k_post -> post_k(k);
}
//================================================================================

// Wishart generator =============================================================
rwish(const df, const Sc)
{
	decl p = sizer(Sc);
	decl B = 0 * Sc;
 
	if(p>1)
	{
		decl i,j;
		for(j=0; j<p; ++j)
		{
			B[j][j] = sqrt( rangamma(1, 1, (df-j)/2, 1/2) );   
		}

		for(i=1; i<p; ++i)
		{
			for(j=0; j<i; ++j)
			{
				B[i][j] = rann(1,1);
			}
		}   
	}
 
	decl A = choleski(Sc);
 
	return A * B * B' * A';
}
//================================================================================

// Simulate zt ===================================================================
sim(const T, const A_true, const k_true, const d_true)
{
	decl A_sim = A_true, i;
	decl q = sizer(A_true);
 
	// Storage	=========================
	decl keep_z = new matrix[T][q];
	//==================================

	decl Q = unit(2);
	decl Qu, Qd, Qv;
         decsvd(Q, &Qu, &Qd, &Qv);

	decl Q_neg_sqrt = Qu * diag(Qd.^(-d_true/2)) * Qv';
	decl S = (1/k_true) * Q_neg_sqrt * A_sim * Q_neg_sqrt;

	decl Pu, Pd, Pv, scale_S, Qinv;
	decl Qinv_star, P, scale_P, z;

	for(i=0; i<T; ++i)
	{
		decl scale_S = choleski(S);
		Qinv = rwish(k_true, S);   

		Q = invert(Qinv);
		decsvd(Q, &Qu, &Qd, &Qv);    
		Q_neg_sqrt = Qu * diag(Qd.^(-d_true/2)) * Qv';
		S = (1/k_true) * Q_neg_sqrt * A_sim * Q_neg_sqrt;
		
		Qinv_star = invert(diag((diagonal(Q).^.5)));
		P         = Qinv_star * Q * Qinv_star;

		decsvd(P, &Pu, &Pd, &Pv);
		scale_P = Pu * diag(Pd.^.5) * Pv';   
		z = (scale_P * rann(q,1))';

		keep_z[i][] = z;
	}

 	return keep_z;
}
//================================================================================


// Set initial values for Qt =====================================================
Initials(const T, const A, const k, const d)
{
	decl q = sizer(A);
 
	// Storages	========================
	decl initial     = new array[T+1];
	initial[0:T]   = zeros(q,q);
	//==================================

	decl Q = unit(q);
	decl Qu, Qd, Qv;
		 decsvd(Q, &Qu, &Qd, &Qv);

	decl Q_neg_sqrt = Qu*diag(Qd.^(-d/2))*Qv';
	decl S = (1/k) * Q_neg_sqrt * A * Q_neg_sqrt;

	initial[0] = Q;
	decl Su, Sd, Sv, Pu, Pd, Pv, Qinv;
	for(decl i=1; i<=T; ++i)
	{
		decl scale_S = choleski(S);
   
		Qinv = scale_S * ranwishart(k, q) * scale_S';  
		Q = invert(Qinv);
   
		decsvd(Q, &Qu, &Qd, &Qv);	   
		Q_neg_sqrt = Qu*diag(Qd.^(-d/2))*Qv';
		S = (1/k) * Q_neg_sqrt * A * Q_neg_sqrt;

		initial[i] = Q;
	}

	return initial;
}
//================================================================================

main()
{
	decl time = timer(), j, t;
	decl n_samples = 1500;
	decl burn      = 500;
 
	decl A_true = invert(reshape(<1, .3, .3, 1>, 2, 2));
 
	//Observations
	decl z = sim(T, A_true, k_true, d_true);
	decl q = sizec(z);
	decl T = sizer(z);
 
	//Store Draws ===========================================
	decl keep_A = new array[n_samples];
		 keep_A[0:(n_samples-1)] = zeros(q,q);

	decl keep_Ainv = new array[n_samples];
		 keep_Ainv[0:(n_samples-1)] = zeros(q,q);
		 
	decl keep_A11     = new matrix[n_samples];
	decl keep_A12     = new matrix[n_samples];
	decl keep_A22     = new matrix[n_samples];
	decl keep_Ainv11  = new matrix[n_samples];
	decl keep_Ainv12  = new matrix[n_samples];
	decl keep_Ainv22  = new matrix[n_samples];	
	decl keep_d       = new matrix[n_samples];
	decl keep_k       = new matrix[n_samples];
	decl A_inv_det    = new matrix[n_samples];
	decl log_Ainv_det = new matrix[n_samples];
	decl log_Q1_det   = new matrix[n_samples];
	decl log_Q2_det   = new matrix[n_samples];
	decl log_Q10_det  = new matrix[n_samples];	
	decl log_QT_det   = new matrix[n_samples];	
	decl acc          = new matrix[T];
	decl Dcorr        = zeros(T,1);
	decl dev          = new matrix[n_samples];

	decl eQe_1   = new matrix[n_samples];
	decl eQe_2   = new matrix[n_samples];
	decl eQe_10  = new matrix[n_samples];	
	decl eQe_500 = new matrix[n_samples];	
	//=======================================================
 
	//Set initial values ====================================
	decl gamma_0 = q;
	decl C0_inv = gamma_0 * unit(q);
	decl lambda0 = .2;
	decl A = unit(q);
	decl k = zeros(1,1); k = 12;
	decl d = zeros(1,1); d = 0.75; 
	decl Qt = Initials(T, A, k, d);
	//========================================================

	// General settings for d and k
	decl ninit      = 4;
	decl dometrop_d = 1;
	decl dometrop_k = 1;
	decl nsamp      = 1;

	//For d
	decl mdl = zeros(1,1); mdl[0][0]     = -0.99;
	decl mdr = zeros(1,1); mdr[0][0]     = 0.99;
	decl dprev = zeros(1,1); dprev[0][0] = 0.75;
	decl samp_d = zeros(1,nsamp); 	
	decl ansd = zeros(1,1);	
	d_post = new postd();	                    

	//For k							
	decl mkl = zeros(1,1); mkl[0][0]     = 2.1;
	decl mkr = zeros(1,1); mkr[0][0]     = 55;
	decl kprev = zeros(1,1); kprev[0][0] = 12;
	decl samp_k = zeros(1,nsamp);
	decl ansk = zeros(1,1);  
	k_post = new postk();
  
	//Start the sampling!
	for(j=0; j<n_samples; ++j)
	{ 
		//Update Qt, t = 1, 2,..., T-1
		decl Qu, Qd, Qv, Qu_can, Qd_can, Qv_can, Q_star_inv, P;  
		decl Q, Qinv, Q_neg_sqrt, S, S_hat, scale_S_hat_L, S_hat_u, S_hat_d, S_hat_v;
		decl canQt, canQtinv, Q_negsqrt_can, S_can, Qinv_star_can, P_can;
		decl Q_cur, Q_negsqrt_cur, Qu_cur, Qd_cur, Qv_cur;
		decl Qinv_star_cur, S_cur, P_cur;
		decl f_Qinv, f_Qinv_val, f_Qinv_c;
		decl Q_sqrt_cur, S_cur_inv, Q_cur_inv, Q_star_cur, P_cur_inv;
   
		for(t=1; t<T; ++t)
		{
			Q = Qt[t-1][][];
			decsvd(Q, &Qu, &Qd, &Qv);
			Q_neg_sqrt = Qu * diag(Qd.^(-d/2)) * Qv';
			S = (1/k) * Q_neg_sqrt * A * Q_neg_sqrt;		
			//S_hat = invert(invert(S) + z[t-1][]' * z[t-1][]);

			if(t==1)
			{
				eQe_1[j] = z[1-1][]*S*z[1-1][]';
			}

			if(t==2)
			{
				eQe_2[j]   = z[2-1][]*S*z[2-1][]';
			}

			if(t==10)
			{
				eQe_10[j]  = z[10-1][]*S*z[10-1][]';
			}

			
			//Sherman-Morrison formula			
			S_hat = S - (1/(1 + z[t-1][]*S*z[t-1][]')) * (S*z[t-1][]'*z[t-1][]*S); 
			
			//Integer-df Wishart
			scale_S_hat_L = choleski(S_hat);		 
			canQtinv = rwish(k+1, S_hat);
			canQt    = invert(canQtinv);

			//Original definition  
			decsvd(canQt, &Qu_can, &Qd_can, &Qv_can);		   //For candidate.
			Q_negsqrt_can = Qu_can * diag(Qd_can.^(-d/2)) * Qv_can';
			S_can         = (1/k) * Q_negsqrt_can * A * Q_negsqrt_can; 	
			Qinv_star_can = diag(diagonal(canQt).^-.5);
			P_can         = Qinv_star_can * canQt * Qinv_star_can;

			f_Qinv        = sqrt(determinant(invert(P_can))) * ( determinant(canQtinv)^((-1-k*d)/2) ) *		 
							exp((-0.5) * trace(invert(S_can) * invert(Qt[t+1][][])) -
							0.5*trace( (invert(P_can)-canQtinv) * z[t-1][]' * z[t-1][]));

			Q_cur         = Qt[t][][];				 
			decsvd(Q_cur, &Qu_cur, &Qd_cur, &Qv_cur);
			Q_negsqrt_cur = Qu_cur * diag(Qd_cur.^(-d/2)) * Qv_cur';
			S_cur         = (1/k) * Q_negsqrt_cur * A * Q_negsqrt_cur;
			Qinv_star_cur = diag(diagonal(Q_cur).^-.5);			   
			P_cur         = Qinv_star_cur * Q_cur * Qinv_star_cur;
		
			f_Qinv_c      = sqrt(determinant(invert(P_cur))) * ( determinant(invert(Q_cur))^((-1-k*d)/2) ) *	  
							exp((-0.5) * trace(invert(S_cur) * invert(Qt[t+1][][])) -
							0.5*trace((invert(P_cur)-invert(Q_cur)) * z[t-1][]' * z[t-1][]));	

			if(j >= burn)
			{
				Dcorr[t-1] = Dcorr[t-1] + P_cur[1][0];
			}
	
			decl R = f_Qinv / f_Qinv_c;
                  
			if(R > ranu(1,1))
			{     
				Qt[t][][] = canQt;
				acc[t-1]  = acc[t-1]+1;
		
				if(j >= burn)
				{
					Dcorr[t-1] = Dcorr[t-1] - P_cur[1][0] + P_can[1][0];
				}
			}
		}

		//Update QT
		decl Qu_T, Qd_T, Qv_T;
		decl ST_can, scale_ST_hat_L, ST_hat, ST_hat_u, ST_hat_d, ST_hat_v;
		decl canQT, canQT_inv, Q_negsqrt_can_T, Qinv_star_can_T, PT_can, f_Qinv_T;
		decl QT_cur, Q_negsqrt_cur_T, Qinv_star_cur_T, PT_cur, f_QinvT, f_QinvT_c;

		if(t==T)
		{
			eQe_500[j] = z[T-1][]*S*z[T-1][]';
		}
		
		//ST_hat = invert(invert(S_cur) + z[T-1][]'*z[T-1][]);		
		//Sherman-Morrison formula   
		ST_hat = S_cur - (1/(1 + z[T-1][]*S_cur*z[T-1][]')) * (S_cur*z[T-1][]'*z[T-1][]*S_cur);
		
		//Integer-df Wishart
		scale_ST_hat_L  = choleski(ST_hat);		
		canQT_inv       = rwish(k+1, ST_hat); 
		canQT           = invert(canQT_inv);

		decsvd(canQT, &Qu_T, &Qd_T, &Qv_T);
		Q_negsqrt_can_T = Qu_T * diag(Qd_T.^(-d/2)) * Qv_T';
		ST_can          = (1/k) * Q_negsqrt_can_T * A * Q_negsqrt_can_T;
		Qinv_star_can_T = diag(diagonal(canQT).^-.5);
		PT_can          = Qinv_star_can_T * canQT * Qinv_star_can_T;

		f_QinvT         = exp( (-0.5)*trace( (invert(PT_can)-canQT_inv) * z[T-1][]' *  z[T-1][] ) );
						  
        //current
		QT_cur          = Qt[T][][];
		Qinv_star_cur_T = diag(diagonal(QT_cur).^-.5);
		PT_cur          = Qinv_star_cur_T * QT_cur * Qinv_star_cur_T;
		
		f_QinvT_c       = exp( (-0.5)*trace( (invert(PT_cur)-invert(QT_cur)) * z[T-1][]' * z[T-1][] ) );

		if(j >= burn)
		{
			Dcorr[T-1] = Dcorr[T-1] + PT_cur[1][0];
		}
		
		decl RT = f_QinvT / f_QinvT_c;
                       
		if(RT > ranu(1,1))
		{                                 
			Qt[T][][] = canQT;
			acc[T-1]  = acc[T-1]+1;

			if(j >= burn)
			{
				Dcorr[T-1] = Dcorr[T-1] - PT_cur[1][0] + PT_can[1][0];
			}
		}

		log_Q1_det[j]   = log(determinant(invert(Qt[1][][])));
		log_Q2_det[j]   = log(determinant(invert(Qt[2][][])));
		log_Q10_det[j]  = log(determinant(invert(Qt[10][][])));		
		log_QT_det[j]   = log(determinant(invert(Qt[T][][])));
		
		//Update A ==================================================================================   
		decl A_inv, Au, Ad, Av, Q_sqrt, scale_A_L, scale_Au, scale_Ad, scale_Av, Cinv_sum=zeros(q,q);
		decl prod_A = new array[T];
			 prod_A[0:T-1] = zeros(q,q);
		
		for(decl t=0; t<T; ++t)
		{
			decsvd(Qt[t][][], &Au, &Ad, &Av);
			Q_sqrt = Au * diag(Ad.^(d/2)) * Av';
			prod_A[t][][] = Q_sqrt * invert(Qt[t+1][][]) * Q_sqrt;
			Cinv_sum += prod_A[t][][];
		}
	 
		decl Cinv = k * Cinv_sum;

		//Integer-df Wishart
		scale_A_L = choleski(invert(C0_inv + Cinv));	 
		A_inv     = rwish((k*T + gamma_0), invert(C0_inv + Cinv));
		A         = invert(A_inv);
	
		A_inv_det[j]     = determinant(A_inv);
		log_Ainv_det[j]  = log(determinant(A_inv));		
		keep_A[j][][]    = A;
		keep_Ainv[j][][] = A_inv;		
		keep_A11[j]      = A[0][0];
		keep_A12[j]      = A[0][1];
		keep_A22[j]      = A[1][1];		
		keep_Ainv11[j]   = A_inv[0][0];
		keep_Ainv12[j]   = A_inv[0][1];
		keep_Ainv22[j]   = A_inv[1][1];
		
		//===========================================================================================

		//Update d ==================================================================================
		decl sum_log_det = 0;			 // to compute psi
		for(decl t=1; t<=T; ++t)
		{
			sum_log_det += log(determinant(invert(Qt[t-1][][])));
		}
		
		decl psi = (-k/2) * sum_log_det;

		d_post -> set_other(psi, A_inv, Qt, k, T, q);
		ansd = Arms_simple(post_d, ninit, mdl, mdr, dometrop_d, dprev, samp_d);
		dprev[0][0] = samp_d[0][0];
		d = samp_d[0][0];
		keep_d[j] = d;
		//===========================================================================================

		//Update k ==================================================================================  
		decl sum_log_Qt = 0;
		decl Cinvk = zeros(q,q);
		decl prod_k = new array[T];
			 prod_k[0:T-1] = zeros(q,q);
	  
		decl Q_sqrt_k, Qu_k, Qd_k, Qv_k;
		for(decl t=0; t<T; ++t)
		{
			decsvd(Qt[t][][], &Qu_k, &Qd_k, &Qv_k);
			Q_sqrt_k = Qu_k * diag(Qd_k.^(d/2)) * Qv_k';
 
			prod_k[t][][] = Q_sqrt_k * invert(Qt[t+1][][]) * Q_sqrt_k;
			sum_log_Qt += log(determinant(prod_k[t][][]));
			Cinvk += prod_k[t][][];
		}

		k_post -> set_other_k(lambda0, A, Cinvk, sum_log_Qt, T, q);
		ansk = Arms_simple(post_k, ninit, mkl, mkr, dometrop_k, kprev, samp_k);
		kprev[0][0] = samp_k[0][0];
		k = samp_k[0][0];
		keep_k[j] = k; 
	}

	decl sum_Ainv = zeros(q,q);
	decl sum_d    = zeros(1,1);
	decl sum_k    = zeros(1,1);
 
	for(decl h = burn; h < n_samples; ++h)
	{
		sum_Ainv += keep_Ainv[h][][];
		sum_d    += keep_d[h];
		sum_k    += keep_k[h];
	}

	print("A_inv = ", (1/(n_samples-burn))*sum_Ainv);
	print("95% Interval for a11*: ", quantilec(keep_Ainv11[burn:(n_samples-1)], <.025, .975>));
	print("95% Interval for a12*: ", quantilec(keep_Ainv12[burn:(n_samples-1)], <.025, .975>));
	print("95% Interval for a22*: ", quantilec(keep_Ainv22[burn:(n_samples-1)], <.025, .975>));
	print("d = ",(1/(n_samples-burn))*sum_d);
	print("95% Interval for d: ", quantilec(keep_d[burn:(n_samples-1)], <.025, .975>));
	print("k = ",(1/(n_samples-burn))*sum_k);
	print("95% Interval for k: ", quantilec(keep_k[burn:(n_samples-1)], <.025, .975>));

	savemat("C:\Research\SPL\Fractional\Ainv11.csv",       keep_Ainv11);
	savemat("C:\Research\SPL\Fractional\Ainv12.csv",       keep_Ainv12);
	savemat("C:\Research\SPL\Fractional\Ainv22.csv",       keep_Ainv22);
	savemat("C:\Research\SPL\Fractional\Ainv_det.csv",     A_inv_det);
	savemat("C:\Research\SPL\Fractional\log_Ainv_det.csv", log_Ainv_det);
	savemat("C:\Research\SPL\Fractional\log_Q1_det.csv",   log_Q1_det);
	savemat("C:\Research\SPL\Fractional\log_Q2_det.csv",   log_Q2_det);
	savemat("C:\Research\SPL\Fractional\log_Q10_det.csv",  log_Q10_det);	
	savemat("C:\Research\SPL\Fractional\log_QT_det.csv",   log_QT_det);	
	savemat("C:\Research\SPL\Fractional\d.csv",            keep_d);
	savemat("C:\Research\SPL\Fractional\k.csv",            keep_k);
	savemat("C:\Research\SPL\Fractional\eQe_1.csv",        eQe_1);
	savemat("C:\Research\SPL\Fractional\eQe_2.csv",        eQe_2);
	savemat("C:\Research\SPL\Fractional\eQe_10.csv",       eQe_10);
	savemat("C:\Research\SPL\Fractional\eQe_500.csv",      eQe_500);
	
	print("T = ", T, "\n");
	print("q = ", q, "\n");
 
	print("time elapsed = ", (timer()-time)/100, "\n");
	print("This is FracWish.", "\n");

	return 0;
}
