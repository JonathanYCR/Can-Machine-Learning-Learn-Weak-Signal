% The Finance-2 application is based on the following paper: 
% Joachim Freyberger & Andreas Neuhierl & Michael Weber, 2017.
% "Dissecting Characteristics Nonparametrically,"
% NBER Working Papers 23227, National Bureau of Economic Research, Inc.
% <https://ideas.repec.org/p/nbr/nberwo/23227.html>
% A copy of the paper can be found in the same folder of this script.
% The original data source is the Center for Research in Security Prices (CRSP) monthly stock file.
% The CRSP data cannot be posted online. The online appendix of Freyberger et al. (2017) describes the construction of the variables.
% The dataset includes firms incorporated in the United States trading on the NYSE, Amex, and Nasdaq with market prices above USD 5.
% The data were provided by Michael Weber. The script below reads the (not posted) original data and transforms them in the format used in the paper.

clear;

load('Data_Imported.mat'); 
load('Y.mat');

% Data are in the long/stacked format. 
% The response variable (Y) is stored in Y.mat 
% The remaining information (X, vrbs) is stored Data_Imported.mat
%    - vrbs contains the Mnemonic of the firm characteristics, which we reproduce below
%    - The first column of X is the identifier of the firm
%    - The second column of X is the month
%    - The remaining columns of Data_Imported are the lagged characteristics
%      The dataset has already the correct temporal structure with the independent variables being lagged characteristics.

vrbs = {
        'a2me',              'assets-to-market cap';                                                                    %Bhandari (1988)
        'at',                'Total assets'        ;                                                                    %Gandhi and Lustig (2015)
        'ato',               'Net sales over lagged net operating assets';                                              %Soliman (2008).
        'beme',              'Ratio of book value of equity to market value of equity';                                 %Rosenberg, Reid, and Lanstein (1985) and Davis, Fama, and French (2000).
        'beta',              'Beta';                                                                                    %Frazzini and Pedersen (2014)
        'c',                 'Ratio of cash and short-term investments to total assets';                                %Palazzo (2012)
        'cto',               'Capital turnover (net sales over lagged total assets)';                                   %Haugen and Baker (1996)
        'd2a',               'Capital intensity (depreciation and amortization over total assets)';                     %Gorodnichenko and Weber (2016).
        'dpi2a',             'Changes in property, plants, and equipment and inventory over lagged total assets';       %Lyandres, Sun, and Zhang (2008)           
        'e2p',               'Earnings to price (income over the market capitalization as of December t-1)';            %Basu (1983)
        'fc2y',              'Fixed costs to sales ';                                                                   %D'Acunto, Liu, Pflueger, and Weber (2016)
        'free_cf',           'Cash flow to book value of equity';                                                       %Hon et al. (2011)).
        'idio_vol',          'Idiosyncratic volatility';                                                                %Ang, Hodrick, Xing, and Zhang (2006).
        'investment',        'Investment (percentage year-on-year growth rate in total assets';                         %Cooper, Gulen, and Schill (2008).)'
        'lev',               'Leverage';                                                                                %Lewellen (2015).'
        'lme',               'Total market capitalization (price times shares outstanding)';                            %Fama and French (1992).'
        'lturnover',         'Turnover (last month volume over shares outstanding';                                     %Datar, Naik, and Radclie (1998))
        'noa',               'Net operating assets';                                                                    %Hirshleifer, Hou, Teoh, and Zhang (2004)          
        'oa',                'Operating accruals';                                                                      %Sloan (1996)
        'ol',                'Operating leverage';                                                                      %Novy-Marx (2011).'
        'pcm',               'price-to-cost margin';                                                                    %Bustamante and Donangelo (2016)
        'pm',                'profit margin';                                                                           %Soliman (2008).'
        'prof',              'Profitability';                                                                           %Ball, Gerakos, Linnainmaa, and Nikolaev (2015)
        'q',                 'Tobin Q'; 
        'rel_to_high_price', 'Closeness to 52-week high (stock price t-1 over the previous 52 week high price)';        %George and Hwang (2004)
        'rna',               'Return on net operating assets';                                                          %Soliman (2008)
        'roa',               'Return on assets';                                                                        %Balakrishnan, Bartov, and Faurel (2010)
        'roe',               'Return on equity';                                                                        %Haugen and Baker (1996)
        'cum_return_12_2',   'Momentum';                                                                                %Fama and French (1996)
        'cum_return_12_7',   'Intermediate momentum';                                                                   %Novy-Marx (2012)
        'cum_return_1_0',    'Short-term reversal ';                                                                    %Jegadeesh (1990)
        'cum_return_36_13',  'Long-term reversal';                                                                      %De Bondt and Thaler (1985).  
        's2p',               'Sales to price';                                                                          %Lewellen (2015)
        'sga2m',             'Ratio of selling, general and administrative expenses to net sales';                      
        'spread_mean',       'Average daily bid-ask spread in the previous months';                                     %Chung and Zhang (2014).'
        'suv',               'Standard unexplained volume (actual volume minus predicted volume in the previous month'  %Garfinkel (2009).
        };

   


Stock_id = X(:,1);
Time = datenum(num2str(X(:,2)), 'yyyymm');
Xraw = X(:, 3:end);
Y = Y;
MnemRaw = vrbs(:,1);
DescRaw = vrbs(:,2);


Missing   =   (sum(isnan([Y Xraw]),2)>0);

Xraw = Xraw(Missing==0,:);
Y = Y(Missing==0,:);
Time = Time(Missing==0);
Stock_id = Stock_id(Missing==0);

Xverylow = Xraw*NaN;
Xlow = Xraw*NaN;
Xveryhigh = Xraw*NaN;
Xhigh = Xraw*NaN;

TimeUnique = unique(Time);

% The code below constructs four dummy variables indicating whether a firm in the previous month 
% belongs to the 1st, 2nd, 4th and 5th quintile of the distribution of that characteristic. 
% We end up with 36x4 independent variables. We do not save the 3th quintile since the constant is included in the regression 

for jt = 1:length(TimeUnique)
    
    TempSel = find(Time==TimeUnique(jt));
    
    for j = 1:length(MnemRaw)
        TempX = Xraw(TempSel,j);        
        Xverylow(TempSel,j)  = (TempX<=quantile(TempX,.20));
            Xlow(TempSel,j)  = (TempX>quantile(TempX,.20) & TempX<=quantile(TempX,.40));
           Xhigh(TempSel,j)  = (TempX>quantile(TempX,.60) & TempX<=quantile(TempX,.80));
        Xveryhigh(TempSel,j) = (TempX>quantile(TempX,.80));     
    end;
    
end;

X = [Xverylow Xlow Xhigh Xveryhigh];

k = length(MnemRaw);
for j = 1:length(MnemRaw)   
    Mnem(j)       =  {[MnemRaw{j},'VeryLow']};
    Mnem(k+j)     =  {[MnemRaw{j},'Low']};
    Mnem(2*k+j)   =  {[MnemRaw{j},'High']};
    Mnem(3*k+j)   =  {[MnemRaw{j},'VeryHigh']};  
end


[x,Mx,Sx] = zscore(X);
[y,My,Sy] = zscore(Y);


save DataWeberQuintiles X Y x y Mx Sx My Sy Mnem Time Stock_id


