% Run SqrtLasso without penalizing components in IND
% but also estimating heteroskedasticity
% psi is associated the initial penalty
function [ betahat, shat ] = MC_TE_SqrtLassoHeteroskedastic_unpenalized ( Y, X, conf_lvl, psi, MaxIt, IND )

[ NumRow, NumCol ] = size(X);
n = NumRow;
m = NumCol;

vv = zeros(NumCol,1);
XX = zeros(NumRow,NumCol);
for j = 1 : NumCol
    vv(j) = norm( X(:,j)/sqrt(n) );
    XX(:,j) = X(:,j) / vv(j) ;
end

                                 % matrix, seed, simulations,
                                 % sample size, quanile
[ lambda ] = MC_TE_SimulateLambdaSqrtLASSO( XX, 1, 2000, n, conf_lvl );
VecLAMBDA = lambda*ones(m,1);
if ( max(size(IND))>0)
    VecLAMBDA(IND) = 0*IND;
end

if (max(size(IND))>0)
    [betaIND, betaIND_INT] = regress(Y, XX(:,IND));
    hatError = (Y - XX(:,IND)*betaIND)*sqrt(n/(n-max(size(IND))));    
else
    hatError = (Y - mean(Y))*sqrt(n/(n-1));
end

Xsq = (XX).^2;

VecLAMBDA = psi*lambda*sqrt(Xsq'*(hatError.^2)/n)/sqrt(sum(hatError.^2/n));
if ( max(size(IND))>0)
        VecLAMBDA(IND) = 0*IND;
end
for K = 1 : MaxIt
    betahat =  SqrtLassoShootingVecLAMBDA(XX,Y,VecLAMBDA);
               
    shat = sum( ( abs(betahat) > 0 ) );

    [ beta2STEP, s2STEP, STDerror2STEP ] = MC_TE_PostEstimator ( Y, XX, betahat, 0, 0 );
    hatError = (Y - XX*beta2STEP)*sqrt(n/(n-s2STEP));
    
    VecLAMBDA = lambda*sqrt(Xsq'*(hatError.^2)/n)/sqrt(sum(hatError.^2/n));
    if ( max(size(IND))>0)
        VecLAMBDA(IND) = 0*IND;
    end
    
end
beta_L1 =  SqrtLassoShootingVecLAMBDA(XX,Y,VecLAMBDA);
betahat = beta_L1 ./ vv;
shat = sum( ( abs(betahat) > 0 ) );
end


function [w,wp,m] = SqrtLassoShootingVecLAMBDA(X, y, lambdaVec,varargin)
% This function computes the SQRT-LASSO estimator
%  We assume that 
%  lambdaVec(j)  = lambda * sqrt( En[ x_{ij}^2 ] )
%
% min  sqrt( Qhat(beta) ) + (lambda/n)*\sum_j sqrt(En[x_{ij}^2])*|beta_j|
%
%
%
[maxIter,verbose,optTol,zeroThreshold] = process_options(varargin,'maxIter',10000,'verbose',0,'optTol',1e-5,'zeroThreshold',1e-4);
[n p] = size(X);

% Start from the Least Squares solution
%MM = eye(p);
%for j = 1 : p 
%    MM(j,j) = lambdaVec(j);
%end
%beta = (X'*X + MM)\(X'*y);
beta = zeros(p,1);

w_old = beta;
k=1;
wp = beta;
    
% Start the log
if verbose==2
    fprintf('%10s %10s %15s %15s %15s\n','iter','shoots','n(w)','n(step)','f(w)');
end

m = 0;

XX = X'*X/n;
Xy = X'*y/n;
while m < maxIter
        
    
    beta_old = beta;
    for j = 1:p
        % lambda without LOADING...
        lambda = lambdaVec(j)/sqrt(XX(j,j));
        
        % Compute the Shoot and Update the variable
        S0 = sum(XX(j,:)*beta) - XX(j,j)*beta(j) - Xy(j);
        ERROR = y - X*beta +  X(:,j)*beta(j);
        Qhat = sum( ERROR.^2 )/n;
        %%% Note that by C-S
        %%% S0^2 <= Qhat * XX(j,j)   :)
        if S0 > (lambda/n) * sqrt(XX(j,j)*Qhat)
            %%% Optimal beta(j) < 0
            %beta(j,1) = (lambda - S0)/XX2(j,j); for LASSO
            % for SQRT-LASSO
            beta(j,1) = (  ( lambda / sqrt( n^2 - lambda^2 ) ) *  sqrt( Qhat * XX(j,j) - S0^2 )  - S0 )   /   XX(j,j);
        elseif S0 < -(lambda/n) * sqrt(XX(j,j)*Qhat)
            %%% Optimal beta(j) > 0
            % beta(j,1) = (-lambda - S0)/XX2(j,j); for LASSO
            % for SQRT-LASSO
            beta(j,1) = (   - ( lambda / sqrt( n^2 - lambda^2 ) ) *  sqrt( Qhat * XX(j,j) - S0^2 )  - S0 )   /   XX(j,j);

        elseif abs(S0) <= (lambda/n) * sqrt(XX(j,j)*Qhat)
            beta(j,1) = 0;
        end
        
    end
    
    m = m + 1;
    
    % Update the log
    if verbose==2
        fprintf('%10d %10d %15.2e %15.2e %15.2e\n',m,m*p,sum(abs(beta)),sum(abs(beta-w_old)),...
            sqrt( sum((X*beta-y).^2)/n )  +  (lambdaVec/n)'*abs(beta) );
        w_old = beta;
        k=k+1;
        wp(:,k) = beta;
    end
    % Check termination
    if sum(abs(beta-beta_old)) < optTol
        break;
    end
    
    
end
if verbose
fprintf('Number of iterations: %d\nTotal Shoots: %d\n',m,m*p);
end
w = beta;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
