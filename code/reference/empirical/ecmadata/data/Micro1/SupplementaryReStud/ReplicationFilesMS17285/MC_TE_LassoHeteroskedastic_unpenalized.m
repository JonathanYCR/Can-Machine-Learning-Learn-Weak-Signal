% Run Lasso without penalizing components in IND
% but also estimating sigma
%  psi is associated the initial penalty
function [ betahat, shat ] = MC_TE_LassoHeteroskedastic_unpenalized ( Y, X, conf_lvl, psi, MaxIt, IND )

[ NumRow, NumCol ] = size(X);
n = NumRow;
m = NumCol;

vv = zeros(NumCol,1);
XX = zeros(NumRow,NumCol);
for j = 1 : NumCol
    vv(j) = norm( X(:,j)/sqrt(n) );
    XX(:,j) = X(:,j) / vv(j) ;
end

                                 % matrix, variance, seed, simulations,
                                 % sample size, quanile
[ lambda ] = MC_TE_SimulateLambda ( XX, 1, 1, 2000, n, conf_lvl );
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

VecLAMBDA = psi*lambda*sqrt(Xsq'*(hatError.^2)/n);
if ( max(size(IND))>0)
        VecLAMBDA(IND) = 0*IND;
end
for K = 1 : MaxIt
    betahat =  LassoShootingVecLAMBDA(XX,Y,VecLAMBDA);
    shat = sum( ( abs(betahat) > 0 ) );

    [ beta2STEP, s2STEP, STDerror2STEP ] = MC_TE_PostEstimator ( Y, XX, betahat, 0, 0 );
    hatError = (Y - XX*beta2STEP)*sqrt(n/(n-s2STEP));
    
    VecLAMBDA = lambda*sqrt(Xsq'*(hatError.^2)/n);
    if ( max(size(IND))>0)
        VecLAMBDA(IND) = 0*IND;
    end
    
end
beta_L1 =  LassoShootingVecLAMBDA(XX,Y,VecLAMBDA);
betahat = beta_L1 ./ vv;
shat = sum( ( abs(betahat) > 0 ) );
end



function [w,wp,m] = LassoShootingVecLAMBDA(X, y, lambdaVec,varargin)
% This function computes the Least Squares parameters
% with a penalty on the L1-norm of the parameters
%
% Method used:
%   The Shooting method of [Fu, 1998]
%
% Modifications:
%   We precompute the Hessian diagonals, since they do not 
%   change between iterations
[maxIter,verbose,optTol,zeroThreshold] = process_options(varargin,'maxIter',10000,'verbose',0,'optTol',1e-5,'zeroThreshold',1e-4);
[n p] = size(X);

% Start from the Least Squares solution
%MM = eye(p);
%for j = 1 : p 
%    MM(j,j) = lambdaVec(j);
%end
%beta = pinv(X'*X + MM)*(X'*y);
beta = zeros(p,1);

% Start the log
w_old = beta;
k=1;
wp = beta;

if verbose==2
    fprintf('%10s %10s %15s %15s %15s\n','iter','shoots','n(w)','n(step)','f(w)');
end

m = 0;

XX2 = X'*X*2;
Xy2 = X'*y*2;
while m < maxIter
    
    
    
    beta_old = beta;
    for j = 1:p
        lambda = lambdaVec(j);
        % Compute the Shoot and Update the variable
        S0 = sum(XX2(j,:)*beta) - XX2(j,j)*beta(j) - Xy2(j);
        if S0 > lambda
            beta(j,1) = (lambda - S0)/XX2(j,j);
        elseif S0 < -lambda
            beta(j,1) = (-lambda - S0)/XX2(j,j);
        elseif abs(S0) <= lambda
            beta(j,1) = 0;
        end
        
    end
    
    m = m + 1;
    
    % Update the log
    if verbose==2
        fprintf('%10d %10d %15.2e %15.2e %15.2e\n',m,m*p,sum(abs(beta)),sum(abs(beta-w_old)),...
            sum((X*beta-y).^2)+lambdaVec'*abs(beta));
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
