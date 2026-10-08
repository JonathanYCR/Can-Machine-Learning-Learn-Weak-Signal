%
%
%


function [ c1, c2, b_int, beta0second, beta0first, var_noise_second, var_noise_first] = MC_TE_GetCoef ( seed_val, rho, alpha0, R21, R22, design, p, n )

SZMat = toeplitz(rho.^(0:(p-1)));

% Let R12 be the R^2 from the regression of the treatment on controls
% Let R22 be the R^2 from the regression of the outcome on controls
% Note that in our setup (with homoskedastic errors),
% R12 = (c1^2)*beta0first'*SZMat*beta0first/(var_noise_first + 
%   (c1^2)*beta0first'*SZMat*beta0first)
% and
% R22 = (c2*beta0second + alpha0*c1*beta0first)'*SZMat*(c2*beta0second +
%   alpha0*c1*beta0first)/(alpha0^2*var_noise_first + var_noise_second +
%   (c2*beta0second + alpha0*c1*beta0first)'*SZMat*(c2*beta0second +
%   alpha0*c1*beta0first))
% c2 = (-b +/- sqrt(b^2 - 4ac))/2a where
% a = (1-R22)*beta0second'*SZMat*beta0second
% b = 2*(1-R22)*alpha0*c1*beta0first'*SZMat*beta0second
% c = (1-R22)*alpha0^2*c1^2*beta0first'*SZMat*beta0first - R22*(alpha0^2*var_noise_first+var_noise_second)


%%%%% Setting the coefficient pattern:
switch( design )
    
    case {1001}
        % constant coefficients
        b_int = 0;
        beta0second = zeros(p,1);
        beta0first  = zeros(p,1);
        beta0second(2:2:40) = 1;
        beta0first(1:2:40) = 1;
        var_noise_second = 1;
        var_noise_first = 1;
        
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    case {1,3,5}
        % case 1 Linear Decay, 
        % case 3 linear decay heteroskedastic
        % case 5 binary treatment
        b_int = 0;
        beta0second = zeros(p,1);
        beta0first  = zeros(p,1);
        beta0second(1:5) = 1./(1:5);
        beta0second(11:15) = 1./(1:5);
        var_noise_second = 1;
        beta0first(1:10) = 1./(1:10);
        var_noise_first  = 1;
        
        %R12 = (beta0first'*SZMat*beta0first ) / ( 1 + beta0first'*SZMat*beta0first)
        %R22 = (beta0second'*SZMat*beta0second ) / ( 1 + beta0second'*SZMat*beta0second)
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        
    case {2,4} % Quadratic Decay
        b_int = 0;
        beta0second = zeros(p,1);
        beta0first  = zeros(p,1);
        beta0second(1:5) = (1./(1:5)).^2;
        beta0second(11:15) = (1./(1:5)).^2;
        var_noise_second = 1;
        beta0first(1:10) = (1./(1:10)).^2;
        var_noise_first  = 1;
        
        %R12 = (beta0first'*SZMat*beta0first ) / ( 1 + beta0first'*SZMat*beta0first)
        %R22 = (beta0second'*SZMat*beta0second ) / ( 1 + beta0second'*SZMat*beta0second)
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
         
    case {22, 44} % All Quadratic Decay
        b_int = 0;
        beta0second = zeros(p,1);
        beta0first  = zeros(p,1);
        beta0second = (1./(1:p)').^2;
        beta0first = (1./(1:p)').^2;
        var_noise_second = 1;
        var_noise_first  = 1;
        
        %R12 = (beta0first'*SZMat*beta0first ) / ( 1 + beta0first'*SZMat*beta0first)
        %R22 = (beta0second'*SZMat*beta0second ) / ( 1 + beta0second'*SZMat*beta0second)
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        

    case {6} % random coef
        randn('state',seed_val);
        A = [1 .8 ; .8 1];
        CA = chol(A);
        aux = randn ( p , 2 )*CA;
        
        b_int = 0;
        beta0second = aux(:,2);
        beta0first  = aux(:,1);
        var_noise_second = 1;
        var_noise_first  = 1;
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
        
        
    case {7}
        randn('state',seed_val);
        A = [1 .8 ; .8 1];
        CA = chol(A);
        aux = randn ( p , 2 )*CA;
        
        b_int = 0;
        beta0second = zeros(p,1);
        beta0first  = zeros(p,1);
        beta0second(1:5) = 1./(1:5);
        beta0second(11:15) = 1./(1:5);
        var_noise_second = 1;
        beta0first(1:10) = 1./(1:10);
        var_noise_first  = 1;
        
        beta0first = beta0first.*aux(:,1);
        beta0second = beta0second.*aux(:,2);
        %R12 = (beta0first'*SZMat*beta0first ) / ( 1 + beta0first'*SZMat*beta0first)
        %R22 = (beta0second'*SZMat*beta0second ) / ( 1 + beta0second'*SZMat*beta0second)
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
        
        
    case {72}
        randn('state',seed_val);
        A = [1 .8 ; .8 1];
        CA = chol(A);
        aux = randn ( p , 2 )*CA;
        
        b_int = 0;
        beta0second = zeros(p,1);
        beta0first  = zeros(p,1);
        beta0second(1:5) = (1./(1:5)).^2;
        beta0second(11:15) = (1./(1:5)).^2;
        var_noise_second = 1;
        beta0first(1:10) = (1./(1:10)).^2;
        var_noise_first  = 1;
        
        beta0first = beta0first.*aux(:,1);
        beta0second = beta0second.*aux(:,2);
        %R12 = (beta0first'*SZMat*beta0first ) / ( 1 + beta0first'*SZMat*beta0first)
        %R22 = (beta0second'*SZMat*beta0second ) / ( 1 + beta0second'*SZMat*beta0second)
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
         
        
    case {722}
        randn('state',seed_val);
        A = [1 .8 ; .8 1];
        CA = chol(A);
        aux = randn ( p , 2 )*CA;
        
        b_int = 0;
        beta0second = zeros(p,1);
        beta0first  = zeros(p,1);
        beta0second = (1./(1:p)').^2;
        beta0first = (1./(1:p)').^2;
        var_noise_second = 1;
        var_noise_first  = 1;
        
        beta0first = beta0first.*aux(:,1);
        beta0second = beta0second.*aux(:,2);
        %R12 = (beta0first'*SZMat*beta0first ) / ( 1 + beta0first'*SZMat*beta0first)
        %R22 = (beta0second'*SZMat*beta0second ) / ( 1 + beta0second'*SZMat*beta0second)
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
       
    case {8}
        randn('state',seed_val);
        A = [1 .8 ; .8 1];
        CA = chol(A);
        aux = randn ( p , 2 )*CA;
        auxU = rand  ( p , 1 );
        
        b_int = 0;
        var_noise_second = 1;
        var_noise_first  = 1;        
        beta0first = aux(:,1).*( 5*(auxU(:,1)<0.05) + 0.05*(auxU(:,1)>=0.05) );
        beta0second = aux(:,2).*( 5*(auxU(:,1)<0.05) + 0.05*(auxU(:,1)>=0.05) );
        %R12 = (beta0first'*SZMat*beta0first ) / ( 1 + beta0first'*SZMat*beta0first)
        %R22 = (beta0second'*SZMat*beta0second ) / ( 1 + beta0second'*SZMat*beta0second)
        c1 = sqrt(R21 / (  (1-R21)*beta0first'*SZMat*beta0first ));
        beta0first  = c1 * beta0first;
        c2 = sqrt(R22 / (  (1-R22)*beta0second'*SZMat*beta0second ));
        beta0second  = c2 * beta0second;
       
end


end

