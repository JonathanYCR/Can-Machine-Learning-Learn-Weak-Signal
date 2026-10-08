%
% Input:
% number of repetitions: NUM_SIM
% sample size: n
% number of variables to be selected: p
% variance of first stage: var_noise1
% variance of second stage: var_noise2
% first stage coefficient intensity:  c1
% second stage coefficient intensity: c2
%
% Output:
% ALL_ALPHA: matrix with all realizations of the NUM_SIM estimates for each of the methods   
%       (1) Lasso
%       (2) post-Lasso
%       (3) sqrt Lasso
%       (4) post-sqrt Lasso
%       (5) indirect post-Lasso (1 aggregated reg)
%       (6) second indirect post-LASSO (reg. Y on d and vars selected by sqrt-Lasso of d on X)
%       (7) new proposal (adding regressors in (6) and (2) using sqrt LASSO)
%       (8) use (7) for estimation but the standard error from (6)
%       (9) double selection method
%           1.  Run d on X, to select X1
%           2.  Run Y on X, tp select X2
%           3.  Run Y on d, X1 and X2.
%       (10) double selection with undersmoothing (same as (9) with lambda = 1/2 ...)
%       (11) double selection method with I3
%       (12) second stage oracle 
%       (13) double selection oracle
%       (14) Split Sample
% ALL_StdErr: the associated standard errors
%
function [ ALL_StdErr, ALL_ALPHA ] = MC_TE_FixedDesign_Heteroskedastic_Lasso ( NUM_SIM, rho, alpha0, R21, R22, design, p, n )


conf_lvl = 0.05; % Confidence level parameter for model selection penalty

psi = 0.75; % initial parameter for heteroskedastic estimation of loadings

ALL_StdErr   = [];
ALL_ALPHA = [];

for k = 1 : NUM_SIM

        %%%%% Setting the coefficient pattern:
        [ c1, c2, b_int, beta0second, beta0first, var_noise_second, var_noise_first] = ...
            MC_TE_GetCoef ( NUM_SIM, rho, alpha0, R21, R22, design, p, n );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
        if floor(k/100) == k/100
            fprintf('Design: %d .  Number of Instance: %d of %d\n', design, k, NUM_SIM);                
        end
        [ Y , X ] = MC_TE_Design_New( k, design, rho, b_int, alpha0, beta0second, beta0first, p, n, var_noise_second, var_noise_first );
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        %% (1) Lasso without penalizing the treatment
        [ betaLASSO, sLASSO ] = MC_TE_LassoHeteroskedastic_unpenalized ( Y, X, conf_lvl, psi, 5, [ 1 2 ] );
        [ suppLASSO ] = MC_TE_GetSupport ( betaLASSO, 0, [ 2 ] );
        [ se_hetLASSO ] = Heteroskedastic_se ( Y, X(:,2), X(:,suppLASSO), betaLASSO(2) );
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %% (2) Post-Lasso (where treatment was not penalized)
        [ beta2STEP, s2STEP, STDerror2STEP ] = MC_TE_PostEstimator ( Y, X, betaLASSO, 0, 2 );
        [ supp2STEP ] = MC_TE_GetSupport ( betaLASSO, 0, [ 2 ] );
        [ se_het2STEP ] = Heteroskedastic_se ( Y, X(:,2), X(:,supp2STEP), beta2STEP(2) );
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% (3) Sqrt-Lasso without penalizing the treatment
        %[ betaSQLASSO, sSQLASSO ] = MC_TE_SqrtLassoHeteroskedastic_unpenalized ( Y, X, conf_lvl, 0.25, 5, [ 1 2 ] );
        %[ suppSQLASSO ] = MC_TE_GetSupport ( betaSQLASSO, 0, [ 2 ] );
        %[ se_hetSQLASSO ] = Heteroskedastic_se ( Y, X(:,2), X(:,suppSQLASSO), betaSQLASSO(2) );
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %% (4) Post-Sqrt-Lasso (where treatment was not penalized)
        %[ beta2STEPsq, s2STEPsq, STDerror_sq ]  = MC_TE_PostEstimator ( Y, X, betaSQLASSO, 0, 2 );
        %[ supp2STEPsq ] = MC_TE_GetSupport ( beta2STEPsq, 0, [ 2 ] );
        %[ se_het2STEPsq ] = Heteroskedastic_se ( Y, X(:,2), X(:,supp2STEPsq), beta2STEPsq(2) );
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% (5) Indirect post-Lasso
        %INDaux = [ 1 (3):1:(2+p) ];
        %[ AGGLASSO, AGGsLASSO ] = MC_TE_LassoHeteroskedastic_unpenalized ( X(:,2), X(:, INDaux), conf_lvl, 0.25, 5, [ 1 ] );
        %[ AGG2STEP, AGGs2STEP, AGGSTDerror ]  = MC_TE_PostEstimator ( X(:,2), X(:, INDaux), AGGLASSO, 0, 0 );       
                        
        
        %AGGvec = [ ones(n,1) X(:,2) - X(:, INDaux)*AGG2STEP ];
        %[AGGpost, AGGpostINT] = regress(Y, AGGvec );
        %AGGmat = inv(AGGvec'*AGGvec);
        %seAGG = sqrt(AGGmat(2,2));
        
        %[ se_hetAGG ] = Heteroskedastic_se ( Y, AGGvec(:,2), AGGvec(:,1), AGGpost(2) );
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% (6) second indirect post Lasso
        %IND_AGG = [];
        %pAGG = max(size(INDaux));
        %for kk = 2 : pAGG 
        %    if ( abs(AGG2STEP(kk)) > 1.0e-8 )
        %        IND_AGG = [ IND_AGG INDaux(kk) ];
        %    end
        %end
        %AGGvecSECOND = X(:, [ 1 2 IND_AGG ]);
        %[AGGpostSECOND, AGGpostINTSECOND] = regress(Y, AGGvecSECOND);
        %AGGmatSECOND = inv(AGGvecSECOND'*AGGvecSECOND);
        %seAGGSECOND = sqrt(AGGmatSECOND(2,2));
        %sSECOND = 2 + max(size(IND_AGG));
   
        %[ se_hetAGGSECONG ] = Heteroskedastic_se ( Y, X(:,2), X(:,[ 1 IND_AGG]), AGGpostSECOND(2) );       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% (7) new proposal (adding regressors in (6) and (2) using sqrt
        %% LASSO)
        %betaRegY = beta2STEP;
        %betaRegD = zeros(1,2+p);
        %betaRegD(1) = AGGLASSO(1);
        %betaRegD(2) = 0;
        %betaRegD(3:(2+p)) = AGGLASSO(2:1:(1+p));
        
        %IND_NEW = [];
        
        %for kk = 3 : 2+p 
        %    if ( abs(betaRegD(kk)) > 1.0e-8 || abs(betaRegY(kk)) > 1.0e-8 )
        %        IND_NEW = [ IND_NEW kk ];
        %    end
        %end
        %NEWvecSECOND = X(:, [ 1 2 IND_NEW ]);
        %[NEWpostSECOND, NEWpostINTSECOND] = regress(Y, NEWvecSECOND);
        %NEWmatSECOND = inv(NEWvecSECOND'*NEWvecSECOND);
        %seNEWSECOND = sqrt(NEWmatSECOND(2,2));
        %sNEWSECOND = 2 + max(size(IND_NEW));
 
        %[ se_hetNEWSECONG ] = Heteroskedastic_se ( Y, X(:,2), X(:,[ 1 IND_NEW]), NEWpostSECOND(2) );
        
        
        %%% (8) is a mix of estimates and se above
        
        %%%%%%% (9) double selection
        %%% 1.  Run D on X, select X1
        %%% 2.  Run Y on X, select X2
        %%% 3.  Run Y on D, X1 and X2.
        [ betaLASSO_1, sLASSO_1] = MC_TE_LassoHeteroskedastic_unpenalized( X(:,2), [ X(:,1) X(:,1) X(:,3:(p+2))], conf_lvl, psi, 5, [ 1 ] );
        [ betaLASSO_2, sLASSO_2] = MC_TE_LassoHeteroskedastic_unpenalized(Y, [ X(:,1) X(:,1) X(:,3:(p+2))],  conf_lvl, psi, 5, [ 1 ] );
                
        IND_NEW = [];
        for kk = 3 : 2+p 
            if ( abs(betaLASSO_1(kk)) > 1.0e-8 || abs(betaLASSO_2(kk)) > 1.0e-8 )
                IND_NEW = [ IND_NEW kk ];
            end
        end
        NEWvecSECOND_9 = X(:, [ 1 2 IND_NEW ]);
        [NEWpostSECOND_9, NEWpostINTSECOND_9] = regress(Y, NEWvecSECOND_9);
        NEWmatSECOND_9 = inv(NEWvecSECOND_9'*NEWvecSECOND_9);
        seNEWSECOND_9 = sqrt(NEWmatSECOND_9(2,2));
        sNEWSECOND_9 = 2 + max(size(IND_NEW));

        [ se_hetNEWSECOND_9 ] = Heteroskedastic_se ( Y, X(:,2), X(:,[ 1 IND_NEW]), NEWpostSECOND_9(2) );
        
        %%%%%%% %%%%%%% (10) double selection with undersmothing ...
        %%% 1.  Run D on X, select X1
        %%% 2.  Run Y on X, select X2
        %%% 3.  Run Y on D, X1 and X2.
        %[ lambda ] = MC_TE_SimulateLambdaSqrtLASSO( [ X(:,1) X(:,1) X(:,3:(p+2))], k, 2000, n, conf_lvl );
        %gamma = zeros(p,1);
        %for j = 1 : (p+2)
        %    gamma(j) =  norm( X(:,j)/sqrt(n) );
        %end
        % we repeated the intercep in the second component for simplicity and did not penalize it 
        %gamma(2) = 0; %gamma(1);%%% Note that int was penalized before 
        %[ betaSQLASSO_1, sSQLASSO_1, lambda_1 ] = MC_TE_SqrtLasso_AddSparse ( X(:,2), [ X(:,1) X(:,1) X(:,3:(p+2))], lambda, gamma, 1.5 );
        %[ betaSQLASSO_2, sSQLASSO_2, lambda_2 ] = MC_TE_SqrtLasso_AddSparse ( Y, [ X(:,1) X(:,1) X(:,3:(p+2))], lambda, gamma, 1.5 );
        %IND_NEW = [];
        %for kk = 3 : 2+p 
        %    if ( abs(betaSQLASSO_1(kk)) > 1.0e-8 || abs(betaSQLASSO_2(kk)) > 1.0e-8 )
        %        IND_NEW = [ IND_NEW kk ];
        %    end
        %end
        %NEWvecSECOND_10 = X(:, [ 1 2 IND_NEW ]);
        %[NEWpostSECOND_10, NEWpostINTSECOND_10] = regress(Y, NEWvecSECOND_10);
        %NEWmatSECOND_10 = inv(NEWvecSECOND_10'*NEWvecSECOND_10);
        %seNEWSECOND_10 = sqrt(NEWmatSECOND_10(2,2));
        %sNEWSECOND_10 = 2 + max(size(IND_NEW));       

        %[ se_hetNEWSECOND_10 ] = Heteroskedastic_se ( Y, X(:,2), X(:,[ 1 IND_NEW]), NEWpostSECOND_10(2) );
        
        %% (11) double selection with I3 = SqrtLasso of Y on d (unp) and x
        %[ betaLASSO_1, sLASSO_1] = MC_TE_LassoHeteroskedastic_unpenalized( X(:,2), [ X(:,1) X(:,1) X(:,3:(p+2))], conf_lvl, 0.25, 5, [] );
        %[ betaLASSO_2, sLASSO_2] = MC_TE_LassoHeteroskedastic_unpenalized(Y, [ X(:,1) X(:,1) X(:,3:(p+2))],  conf_lvl, 0.25, 5, [] );
                
        IND_NEW = [];
        for kk = 3 : 2+p 
            if ( abs(betaLASSO_1(kk)) > 1.0e-8 || abs(betaLASSO_2(kk)) > 1.0e-8 || abs(betaLASSO(kk)) > 1.0e-8 )
                IND_NEW = [ IND_NEW kk ];
            end
        end
        NEWvecSECOND_11 = X(:, [ 1 2 IND_NEW ]);
        [NEWpostSECOND_11, NEWpostINTSECOND_11] = regress(Y, NEWvecSECOND_11);
%         NEWmatSECOND_11 = inv(NEWvecSECOND_11'*NEWvecSECOND_11);
%         seNEWSECOND_11 = sqrt(NEWmatSECOND_11(2,2));
        sNEWSECOND_11 = 2 + max(size(IND_NEW));
        
        [ se_hetNEWSECOND_11 ] = Heteroskedastic_se ( Y, X(:,2), X(:,[ 1 IND_NEW]), NEWpostSECOND_11(2) );
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% (12) Oracle second stage equation
        ORACLEsecond = abs(beta0second) >= sqrt(var_noise_second/n);
        NEWvecORACLE = X(:, logical([ 1 ; 1 ; ORACLEsecond ]));
        if size(NEWvecORACLE(:,2)) > n
            NEWvecORACLE = NEWvecORACLE(:,1:n-2);
        end
        [NEWpostORACLE, NEWpostINTORACLE] = regress(Y, NEWvecORACLE);
        NEWmatORACLE = inv(NEWvecORACLE'*NEWvecORACLE);
        seNEWORACLE = sqrt(NEWmatORACLE(2,2));
        sNEWORACLE =  2 + sum(ORACLEsecond);       
        
        [ se_hetNEWORACLE ] = Heteroskedastic_se ( Y, X(:,2), X(:,logical([ 1 ; 0 ; ORACLEsecond ])), NEWpostORACLE(2) );
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% (13) Oracle double selection 
        ORACLEfirst = abs(beta0first) >= sqrt(var_noise_second/n);
        ORACLEdouble = max(ORACLEsecond,ORACLEfirst);
        NEWvecORACLEds = X(:, logical([ 1 ; 1 ; ORACLEdouble ]));
        if size(NEWvecORACLEds(:,2)) > n
            NEWvecORACLEds = NEWvecORACLEds(:,1:n-2);
        end
        [NEWpostORACLEds, NEWpostINTORACLEds] = regress(Y, NEWvecORACLEds);
        NEWmatORACLEds = inv(NEWvecORACLEds'*NEWvecORACLEds);
        seNEWORACLEds = sqrt(NEWmatORACLEds(2,2));
        sNEWORACLEds = 2 + sum(ORACLEdouble);  
        
        [ se_hetNEWORACLEds ] = Heteroskedastic_se ( Y, X(:,2), X(:,logical([ 1 ; 0 ; ORACLEdouble ])), NEWpostORACLEds(2) );

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% (14) Split Sample
        [ alpha_ss, se_ss ] = MC_TE_LassoHeteroskedastic_SplitSample ( Y, X, conf_lvl, k );
        
        
        %%%% INFERENCE SUMMARY %%%%%%%%%%%%%%%%%%%
        ALL_StdErr = [ALL_StdErr ; se_hetLASSO ...
                                   se_het2STEP ... 
                                   1 ... %se_hetSQLASSO ...
                                   1 ... %se_het2STEPsq ...
                                   1 ... %se_hetAGG ...
                                   1 ... %se_hetAGGSECONG ...
                                   1 ... %se_hetNEWSECONG ...
                                   1 ... %se_hetAGGSECONG ...
                                   se_hetNEWSECOND_9 ...
                                   1 ... %se_hetNEWSECOND_10 ...
                                   se_hetNEWSECOND_11 ...
                                   se_hetNEWORACLE ... 
                                   se_hetNEWORACLEds ...
                                   se_ss];
        ALL_ALPHA = [ ALL_ALPHA ; betaLASSO(2) ...
                                  beta2STEP(2) ...
                                  0 ... %betaSQLASSO(2) ...
                                  0 ... %beta2STEPsq(2)...  
                                  0 ... %AGGpost(2) ...
                                  0 ... %AGGpostSECOND(2)... 
                                  0 ... %NEWpostSECOND(2) ...
                                  0 ... %NEWpostSECOND(2) ...
                                  NEWpostSECOND_9(2) ...
                                  0 ... %NEWpostSECOND_10(2)...
                                  NEWpostSECOND_11(2)...
                                  NEWpostORACLE(2) ...
                                  NEWpostORACLEds(2) ...
                                  alpha_ss ];

end


