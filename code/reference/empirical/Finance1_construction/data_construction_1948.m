clear,clc;

[a, b, c] = xlsread('PredictorData2021.xlsx', 'Annual');

raw_titles = b(1,2:end);
all_years = datetime(num2str(a(:, 1)),'InputFormat', 'yyyy'); 

data = {};

%%% Getting existing series
for i = 1:length(raw_titles)
    temp = a(:, i+1);
    idx = find(~isnan(temp), 1, 'first')
    data.(raw_titles{i}).values = temp(idx:end, 1);
    data.(raw_titles{i}).dates = all_years(idx:end, 1);
end


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

IndexDiv = data.Index.values + data.D12.values;
logrediv = [log(IndexDiv(2:end)) - log(data.Index.values(1:end-1))];
logRfree = log(data.Rfree.values + 1);
re_div = logrediv - logRfree(2:end);
re_div=re_div(77:end)
%re_div=data.CRSP_SPvw.values(23:end)-data.Rfree.values(78:end)
%%% Mnemonics
Mnem = {'dfy', 'infl', 'svar', 'd_e', 'lty', 'tms', 'tbl', 'dfr'...
        'd_p', 'd_y', 'ltr', 'e_p', 'b_m', 'ik', 'ntis', 'eqis'};
              
dataset = [year(all_years(78:end)) re_div]; % 1948 - 2021 for y
for r = 1:length(Mnem)
    idx = find(year(data.(Mnem{r}).dates) == 1947);
    dataset = [dataset data.(Mnem{r}).values(idx:end-1)];
end

Time = dataset(:, 1);
Y = dataset(:, 2);
X = dataset(:, 3:end);
y = zscore(Y);
x = zscore(X);

save('Goyal.mat', 'Time', 'Y', 'X', 'Mnem', 'y', 'x');
