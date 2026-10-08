%
% Gets the support but skips components in the list SKIP
function [ ind_col ] = MC_TE_GetSupport ( beta_FS, X,  SKIP )

SizeSkip  = max(size(SKIP));
p = max(size(beta_FS));

ind = ( abs(beta_FS) > 0 );

ind_col = [];

for j = 1 : p 
    if ( ind(j) == 1 )
        SkipThis = 0;
        for k = 1:SizeSkip
            if  ( SKIP(k) == j )
                SkipThis = 1;
            end
        end
        if ( SkipThis == 0 )
            ind_col = [ ind_col j ];
        end
    end
end

