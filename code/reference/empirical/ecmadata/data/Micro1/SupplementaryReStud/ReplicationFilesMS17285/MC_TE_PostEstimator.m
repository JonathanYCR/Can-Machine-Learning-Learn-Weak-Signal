%
%
%  Post estimator Trimms component j if \hat\sigma_j \hat\beta_j < eps
%
%  CompStdErr: is the component for which we want a std error
function [ betatilde, stilde, STDerror ] = MC_TE_PostEstimator ( Y, X, beta_FS, eps, CompStdErr )

[ n , p ] = size(X);

v = zeros(p,1);
for j = 1 : p 
    v(j) = norm( X(:,j) / sqrt(n) );
end

ind = ( abs(beta_FS.*v) > eps );
if( min(X(:,1))==max(X(:,1)) )
    ind(1) = 1;
end
XX = [];
ind_col = [];

for j = 1 : p 
    if ( ind(j) == 1 )
        XX = [ XX  X(:,j) ];
        ind_col = [ ind_col j ];
    end
end

if size(XX,2) > 1
    if sum(abs(XX(:,1)-XX(:,2))) == 0
        XX(:,2) = [];
        ind_col(2) = [];
    end
end

K = max(size(ind_col));
betatilde = zeros(p,1);

if     ( K >= n  )
    Minv = pinv(XX'*XX);
    
elseif ( K > 0  )
    
    if rank(XX) < size(XX,2),
        disp('hi there');
    end

    Minv = inv(XX'*XX);
        
end

if ( K > 0 )
    betatilde(ind_col) = Minv*XX'*Y;
end

stilde = K;

if (CompStdErr > 0 && CompStdErr <= p )
    STDerror = sqrt( Minv(CompStdErr,CompStdErr) );
else
    STDerror = 0;
end


