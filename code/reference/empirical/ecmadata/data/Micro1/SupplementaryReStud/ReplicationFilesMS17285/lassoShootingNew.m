% Lasso with shooting algorithm
% Author: Xiaohui Chen (xiaohuic@ece.ubc.ca)
% Version: 2012-Feb

function b = lassoShooting(X, Y, lambda, maxIt, tol, standardize);

% Implement the shooting algorithm for the lasso in the penalized form:
% minimize .5*||Y-X*beta||_2^2 + lambda*||beta||_1.
% Ref: Fu (1998) Penalized regression: the bridge versus the lasso. J. Comput.
% Graph. Stats.
% Note: suitable for the under-determined case.
% Caution: slow for large scale problem.

if nargin < 6, standardize = true; end
if nargin < 5, tol = 1e-10; end
if nargin < 4, maxIt = 1e4; end

if standardize,
    X = normalize(X);
    Y = Y-mean(Y);
end

% Initialization
[n,p] = size(X);
if p > n,
    b = zeros(p,1); % From the null model, if p > n
else
    b = X \ Y;  % From the OLS estimate, if p <= n
end
b_old = b;
i = 0;

% Precompute X'X and X'Y
XTX = X'*X;
XTY = X'*Y;

% Shooting loop
while i < maxIt,
    i = i+1;
    for j = 1:p,
        jminus = setdiff(1:p,j);
        S0 = XTX(j,jminus)*b(jminus) - XTY(j);  % S0 = X(:,j)'*(X(:,jminus)*b(jminus)-Y)
        if S0 > lambda,
            b(j) = (lambda-S0) / norm(X(:,j),2)^2;
        elseif S0 < -lambda,
            b(j) = -(lambda+S0) / norm(X(:,j),2)^2;
        else
            b(j) = 0;
        end
    end
    delta = norm(b-b_old,2);    % Norm change during successive iterations
    if delta < tol, break; end
    b_old = b;
end
if i == maxIt,
    fprintf('%s\n', 'Maximum number of iteration reached, shooting may not converge.');
end


% Normalize columns of X to have mean zero and length one.
function sX = normalize(X)

[n,p] = size(X);
sX = X-repmat(mean(X),n,1);
sX = sX*diag(1./sqrt(ones(1,n)*sX.^2));