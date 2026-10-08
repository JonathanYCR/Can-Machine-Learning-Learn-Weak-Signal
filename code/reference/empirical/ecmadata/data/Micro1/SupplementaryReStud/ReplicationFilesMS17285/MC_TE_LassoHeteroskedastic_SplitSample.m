
function [ alpha_ss, se_ss ] = MC_TE_LassoHeteroskedastic_SplitSample ( Y, X, conf_lvl, RndSeed )

[ n , pp ] = size(X);
p = pp-2;         
%% Split Sample
PermutedIndices = randperm(n);
na =floor(n/2);
nb =n-na;
SampleA = PermutedIndices(1:na);
SampleB = PermutedIndices(1+nb:n);
%%

psi = 0.75;

%% Sample A Model Selection (double seleciton)
[ betaLASSO_1, sLASSO_1] = MC_TE_LassoHeteroskedastic_unpenalized( X(SampleA,2), [ X(SampleA,1) X(SampleA,1) X(SampleA,3:(p+2))], conf_lvl, psi, 5, [] );
[ betaLASSO_2, sLASSO_2] = MC_TE_LassoHeteroskedastic_unpenalized(Y(SampleA), [ X(SampleA,1) X(SampleA,1) X(SampleA,3:(p+2))],  conf_lvl, psi, 5, [] );
                
IND_NEW_A = [];
for kk = 3 : 2+p 
            if ( abs(betaLASSO_1(kk)) > 1.0e-8 || abs(betaLASSO_2(kk)) > 1.0e-8 )
                IND_NEW_A = [ IND_NEW_A kk ];
            end
end
%% 

%% Sample B Model Selection (double seleciton)
[ betaLASSO_1, sLASSO_1] = MC_TE_LassoHeteroskedastic_unpenalized( X(SampleB,2), [ X(SampleB,1) X(SampleB,1) X(SampleB,3:(p+2))], conf_lvl, psi, 5, [] );
[ betaLASSO_2, sLASSO_2] = MC_TE_LassoHeteroskedastic_unpenalized(Y(SampleB), [ X(SampleB,1) X(SampleB,1) X(SampleB,3:(p+2))],  conf_lvl, psi, 5, [] );
                
IND_NEW_B = [];
for kk = 3 : 2+p 
            if ( abs(betaLASSO_1(kk)) > 1.0e-8 || abs(betaLASSO_2(kk)) > 1.0e-8 )
                IND_NEW_B = [ IND_NEW_B kk ];
            end
end
%% Cross Estimation
XbhatIa = X(SampleB, [ 1 2 IND_NEW_A ]);
[beta_bhatIa, int_beta_bhatIa] = regress(Y(SampleB), XbhatIa);
XXbhatIa_inv = inv(XbhatIa'*XbhatIa);
Upsilon_b = XXbhatIa_inv(2,2);
s_bhatIa = 2 + max(size(IND_NEW_A));

XahatIb = X(SampleA, [ 1 2 IND_NEW_B ]);
[beta_ahatIb, int_beta_ahatIb] = regress(Y(SampleA), XahatIb);
XXahatIb_inv = inv(XahatIb'*XahatIb);
Upsilon_a = XXahatIb_inv(2,2);
s_ahatIb = 2 + max(size(IND_NEW_B));

%% Combining estimators
alpha_ss = ( (na/n)*Upsilon_a  + (nb/n)*Upsilon_b  )^(-1)*( (na/n)*Upsilon_a*beta_ahatIb(2) + (nb/n)*Upsilon_b*beta_bhatIa(2));

hat_zeta_a = (Y(SampleA) - XahatIb*beta_ahatIb)*sqrt(na/[na-s_ahatIb-1]); 
hat_zeta_b = (Y(SampleB) - XbhatIa*beta_bhatIa)*sqrt(nb/[nb-s_bhatIa-1]); 

beta_bhatIa = regress( X(SampleB,2), X(SampleB,[1 IND_NEW_A]) );
beta_ahatIb = regress( X(SampleA,2), X(SampleA,[1 IND_NEW_B]) );
hat_v_a = X(SampleA,2) - X(SampleA,[1 IND_NEW_B])*beta_ahatIb;
hat_v_b = X(SampleB,2) - X(SampleB,[1 IND_NEW_A])*beta_bhatIa;

%Homoskedastic
%se_ss = norm([ hat_zeta_a ; hat_zeta_b ]/sqrt(n),2)*sqrt( inv( 1/Upsilon_a  + 1/Upsilon_b ) );

%Heteroskedatic
se_ss = sqrt( ...
            ( (hat_zeta_a.^2)'*(hat_v_a.^2) + ...
              (hat_zeta_b.^2)'*(hat_v_b.^2) ) / ...
           (hat_v_a'*hat_v_a + hat_v_b'*hat_v_b)^2    );

