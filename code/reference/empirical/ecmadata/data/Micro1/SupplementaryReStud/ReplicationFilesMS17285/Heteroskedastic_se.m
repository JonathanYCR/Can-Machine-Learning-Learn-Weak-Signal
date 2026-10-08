%
%
%
function [ se_het ] = Heteroskedastic_se ( Y, D, Z, alpha )

% ZtZinv = inv(Z'*Z);
% PZ = Z*ZtZinv*Z';

if ~isempty(Z)
    [ n dimZ] = size(Z);
    Dtilde = D - Z*(Z\D); %d~ = d-z*(z*'z*)^(-1)(z*'d)
    Ytilde = Y - Z*(Z\Y); %y~ = y-z*(z*'z*)^(-1)(z*'y)
else
    n = size(D,1);
    dimZ = 0;
    Dtilde = D;
    Ytilde = Y;
end

Etilde = Ytilde - Dtilde * alpha; %e~ = y~ - d~*alpha

P = (Dtilde.^2)/(Dtilde'*Dtilde);
Estar = Etilde./(1-P);

% se_het = sqrt( (n/(n-dimZ-1))* (Dtilde.^2)'*(Etilde.^2) / (Dtilde'*Dtilde)^2 ); %(n/(n-dim(z*)-1))*sum(d~_i2 e~_i2)/(sum(d~_i2))2
se_het = sqrt( (n/(n-dimZ-1))* ((n-1)/n)* ...
    ((Dtilde.^2)'*((Estar.^2)) / (Dtilde'*Dtilde)^2 ...
    - (1/n)*((Dtilde'*Estar)^2)/(Dtilde'*Dtilde)^2 ));

end
