%  The lambda( 1- ALPHA | X ) is the 1-ALPHA quantile of
%  
%    || S ||_{\infty} =  max_{j <= p} | n^{1/2} E_n[ a_t x_{tj} / s_j ] |
%    s_j = sqrt{ E_n[ x_{tj}^2 ] }
%    a_t = e_t/ sqrt( En[e_i^2] )
%

function [ lambda_final ] = MC_TE_SimulateLambdaSqrtLASSO ( X, seed, NumSim, n, ALPHA )

NumSim = max(NumSim, n);

[ Numrows, NumColumns ] = size( X );

NormXX = zeros(NumColumns,1);
lambda = zeros(NumSim,1);

for j = 1 : NumColumns
    NormXX(j) = norm(X(:,j)/sqrt(n));
end    


for k = 1 : NumSim  
    randn('state', seed+k);
    error = randn(n,1);          
    NormERROR = norm(error)/sqrt(n);
    aa = error/NormERROR;
    lambda(k) = max( abs(  ( X'*aa )./(NormXX) ) );        
end

lambda_final = quantile(lambda, 1-ALPHA ); 

