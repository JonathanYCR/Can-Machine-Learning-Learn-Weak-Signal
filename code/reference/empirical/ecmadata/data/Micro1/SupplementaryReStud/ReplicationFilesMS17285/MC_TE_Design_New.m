%%% 
% Input:
%   seed_val: MC seed;
%   rho: correlation between technical regressors;
%   b_int: coefficient of intersept
%   alpha0: coefficient of treatment;
%   beta0secondstage: coefficient of technical controls in the second statge
%   beta0firststage: coefficient of technical controls in the second statge
%   p: number of technical controls
%   n: sample size
%   var_noise_second: variance of noise in the second stage;
%   var_noise_first: variance of noise in the first stage;
%
% Output:
%    X = [ 1 ; d ; XX ] where:
%               1 is the intercept, 
%               d is hte treatment, 
%               XX contains the technical controls
%    Y response variable
function [ Y , X ] = MC_TE_Design_New( seed_val, design, rho, b_int, alpha0, beta0secondstage, beta0firststage, p, n, var_noise_second, var_noise_first )

randn('state',seed_val);
aux = randn ( n , 1+p+4 );
    
X(:,1) = ones(n,1);
X(:,3) = aux(:,1);
for j = 4 : (p+1+1)
	X(:,j) = rho*X(:,j-1) + sqrt(1-rho^2)*aux(:,j-2);
end
  

switch (design)

    case {3} % heteroskedastic
    
    beta0second = zeros(p,1);
    beta0first  = zeros(p,1);
    beta0second(1:5) = 1./(1:5);
    beta0second(11:15) = 1./(1:5);
    beta0first(1:10) = 1./(1:10);
    
    stdfirst = abs(var_noise_first + X(:,3:p+2)*beta0first);
    stdfirst = sqrt(stdfirst.^2/mean(stdfirst.^2));
    stdsecond = abs(var_noise_second + X*[ b_int ; alpha0 ; beta0second ]);
    stdsecond = sqrt(stdsecond.^2/mean(stdsecond.^2));
    
    %%% Generating Error for Response Variable
    error = stdsecond.*aux(:,1+p+4);
    %%% Generating Error for d Variable
    error_u = stdfirst.*aux(:,1+p+3);

    X(:,2) = X(:,3:p+2)*beta0firststage + error_u;
    
    Y = X*[ b_int ; alpha0 ; beta0secondstage ] + error;

    case {4} % heteroskedastic
    
    beta0second = zeros(p,1);
    beta0first  = zeros(p,1);
    beta0second(1:5) = (1./(1:5)).^2;
    beta0second(11:15) = (1./(1:5)).^2;
    beta0first(1:10) = (1./(1:10)).^2;

    stdfirst = abs(var_noise_first + X(:,3:p+2)*beta0first);
    stdfirst = sqrt(stdfirst.^2/mean(stdfirst.^2));
    stdsecond = abs(var_noise_second + X*[ b_int ; alpha0 ; beta0second ]);
    stdsecond = sqrt(stdsecond.^2/mean(stdsecond.^2));
    
    %%% Generating Error for Response Variable
    error = stdsecond.*aux(:,1+p+4);
    %%% Generating Error for d Variable
    error_u = stdfirst.*aux(:,1+p+3);

    X(:,2) = X(:,3:p+2)*beta0firststage + error_u;
    
    Y = X*[ b_int ; alpha0 ; beta0secondstage ] + error;
    

    case {44} % heteroskedastic
    
    beta0second = zeros(p,1);
    beta0first  = zeros(p,1);
    beta0second = (1./(1:p)').^2;
    beta0first = (1./(1:p)').^2;

    stdfirst = abs(var_noise_first + X(:,3:p+2)*beta0first);
    stdfirst = sqrt(stdfirst.^2/mean(stdfirst.^2));
    stdsecond = abs(var_noise_second + X*[ b_int ; alpha0 ; beta0second ]);
    stdsecond = sqrt(stdsecond.^2/mean(stdsecond.^2));
    
    %%% Generating Error for Response Variable
    error = stdsecond.*aux(:,1+p+4);
    %%% Generating Error for d Variable
    error_u = stdfirst.*aux(:,1+p+3);

    X(:,2) = X(:,3:p+2)*beta0firststage + error_u;
    
    Y = X*[ b_int ; alpha0 ; beta0secondstage ] + error;
    

    
    case {5} % binary treatment
    %%% Generating Error for Response Variable
    error = sqrt(var_noise_second)*aux(:,1+p+4);
    %%% Generating Error for d Variable
    error_u = sqrt(var_noise_first)*aux(:,1+p+3);

    X(:,2) = ( X(:,3:p+2)*beta0firststage + error_u > 0 );
    
    Y = X*[ b_int ; alpha0 ; beta0secondstage ] + error;
    
    
    otherwise
    %%% Generating Error for Response Variable
    error = sqrt(var_noise_second)*aux(:,1+p+4);
    %%% Generating Error for d Variable
    error_u = sqrt(var_noise_first)*aux(:,1+p+3);

    X(:,2) = X(:,3:p+2)*beta0firststage + error_u;
    
    Y = X*[ b_int ; alpha0 ; beta0secondstage ] + error;

end



