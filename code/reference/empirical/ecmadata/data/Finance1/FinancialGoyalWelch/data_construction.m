clear,clc;

[a, b, c] = xlsread('PredictorData.xlsx', 'Annual');

raw_titles = b(1,2:end);
all_years = datenum(num2str(a(:, 1)), 'yyyy'); 

data = {};

%%% Getting existing series
for i = 1:length(raw_titles)
    temp = a(:, i+1);
    idx = find(~isnan(temp), 1, 'first')
    data.(raw_titles{i}).values = temp(idx:end, 1);
    data.(raw_titles{i}).dates = all_years(idx:end, 1);
end

data.ntis.values = data.ntis.values(2:end);
data.ntis.dates = data.ntis.dates(2:end);

data.infl.values = data.infl.values(7:end);
data.infl.dates = data.infl.dates(7:end);

%%% Constructing new series
% d_p: Dividend Price Ratio
data.d_p.values = log(data.D12.values) - log(data.Index.values);
data.d_p.dates = data.D12.dates;

% d_y: Dividend Yield
data.d_y.values = log(data.D12.values(2:end, 1))-log(data.Index.values(1:end-1, 1));
data.d_y.dates = data.D12.dates(2:end, 1);

% e_p: Earnings Price Ratio
data.e_p.values = log(data.E12.values) - log(data.Index.values);
data.e_p.dates = data.E12.dates;

% d_e: Dividend Payout Ratio
data.d_e.values = log(data.D12.values) - log(data.E12.values);
data.d_e.dates = data.D12.dates;

% tms: The Term Spread
data.tms.values = data.lty.values(2:end) - data.tbl.values;
data.tms.dates = data.tbl.dates;


% dfy: Default Yield Spread
data.dfy.values = data.BAA.values - data.AAA.values;
data.dfy.dates = data.BAA.dates;

% dfr: Default Return Spread
data.dfr.values = data.corpr.values -  data.ltr.values;
data.dfr.dates = data.corpr.dates;

% eqp: Equity Premium
IndexDiv = data.Index.values + data.D12.values;
logrediv = [log(IndexDiv(2:end)) - log(data.Index.values(1:end-1))];
logRfree = log(data.Rfree.values + 1);
re_div = logrediv - logRfree(2:end);
display(mean(re_div(1:134)));
display(std(re_div(1:134)));

%%% Mnemonics
all_predictors = {'dfy', 'infl', 'svar', 'd_e', 'lty', 'tms', 'tbl', 'dfr'...
                   'd_p', 'd_y', 'ltr', 'e_p', 'b_m', 'ik', 'ntis', 'eqis'};
              
dataset = [all_years(2:end) re_div];
for r = 1:length(all_predictors)
    idx = find(all_years == data.(all_predictors{r}).dates(1));
    if idx == 1
        temp = [];
    else
        temp = repmat(nan, idx-1, 1);
    end
    dataset = [dataset [temp; data.(all_predictors{r}).values(1:end-1)]];
end

% save all R-square in this structure.
output = {};
%%% Start In-sample regression here
for r = 1:length(all_predictors)
    fit = fitlm(dataset(:,r+2), dataset(:,2));
    output{r, 1} = all_predictors{r};
    output{r, 2} = fit.Rsquared.Adjusted * 100;
end
