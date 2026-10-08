clear 

% generate_freddataGLP.m
% =========================================================================
% DESCRIPTION: 
% This script loads in raw data from the 2016-04 monthly vintage CSV file,
% transforms each series based on transformation code using
% prepare_missing.m, and removes outliers from the transformed data using
% remove_outliers.m.
%
%
% =========================================================================
% CLEAR: 
clear all
close all
clc

% =========================================================================
% PARAMETER TO BE CHANGED:
% Update the .csv filename to match the desired version

% CSV file name
csv_in='2023-12.csv';
%csv_in='2016-03.csv'
% =========================================================================
% LOAD AND LABEL DATA: 
% Load data from CSV file
dum=importdata(csv_in,',');

% Variable names
names=dum.textdata(1,2:end);

% Transformation numbers
tcode=dum.data(1,:);
%datetime(dum.textdata(end,1),'InputFormat', 'MM/dd/yyyy')
% Raw data
rawdata=dum.data(2:end,:);

% Month of final observation
final_month=month(datetime(dum.textdata(end,1),'InputFormat', 'MM/dd/yyyy'));

% Year of final observation
final_year=year(datetime(dum.textdata(end,1),'InputFormat', 'MM/dd/yyyy'));

% =========================================================================
% SET UP DATES: 
% Dates (monthly) are of the form YEAR+MONTH/12
% e.g. March 1970 is represented as 1970+3/12
% Dates go from 1959:01 to final_year:final_month (see above)
dates = [1959+1/12:1/12:final_year+final_month/12]';

time = datetime(dum.textdata(3:end,1),'InputFormat','MM/dd/yyyy');

% T = number of months in sample
T=size(dates,1);
rawdata=rawdata(1:T,:);

% =========================================================================
% TRANSFORM RAW DATA INTO STATIONARY FORM: 
% Use function prepare_missing.m
%   Output yt: matrix containing data after transformation
yt=prepare_missing(rawdata,tcode);

% =========================================================================
% REDUCE SAMPLE TO USABLE DATES: 
% Remove first two months because some series have been second differenced
yt=yt(3:T,:);
dates=dates(3:T,:);
time = time(3:T,:);
% =========================================================================
% REMOVE OUTLIERS: 
% Use function remove_outliers.m (see for definition of outliers)
%   Output data: matrix containing transformed series after removal of outliers
%   Output n: matrix containing number of outliers removed for each series
[data,n]=remove_outliers(yt);


% =========================================================================
% SAVE DATA TO .MAT FILE: 
% Save data, dates, names, and tcode to file freddata.mat
%unbal = names([64, 66, 101,  130, 135])';
unbal = names([ 58 ,60, 79,87, 95,123,127])';

medium = {
    'RPI'
    'DPCERA3M086SBEA'
    'INDPRO'
    'CUMFNS'
    'UNRATE'
    'PAYEMS'
    'HOUST'
    'NAPM'
    'S&P 500'
    'FEDFUNDS'
    'T10YFFM'
    'AAAFFM'
    'BAAFFM'
    'WPSFD49207'
    'WPSID62'
    'OILPRICEx'
    'CPIAUCSL'
    'PCEPI'
    'CES0600000008'
    'MZMSL'};

idMedium = find(ismember(names,medium));
idLarge  = find(ismember(names,unbal)==0);


save FredMDall yt data time dates names tcode unbal medium 
%yt: transformed data
%data: outliers removed
%time: dates in matlab format
%dates: dates in numeric format
%names: mnemonics
%tcod: transformation co
%remove: variables to remove to obtain a balance dataset
%Medium: variables for a medium size model

Target  = 'INDPRO';
Target2= 'CPIAUCSL';
Target3= 'UNRATE';
hor = 1;


StartSample = min(find(year(time)==1960));
  EndSample = max(find(year(time)==2021));

Time = time(StartSample:EndSample);


% idMedium = find(ismember(names,medium));
% Data = yt(StartSample:EndSample,idMedium);
% Mnem = names(idMedium);
% Tcode = tcode(idMedium);
% 
% 
% idTarget  = find(ismember(Mnem,Target));
% YY = filter(ones(1,hor)/hor,1,Data(:,idTarget)); YY = YY(hor+1:end,:);
% XX = Data(1:end-hor,:);
% MY = mean(YY);
% SY =  std(YY);
% 
% MX = mean(XX);
% SX =  std(XX);
% 
% [T,k] = size(XX);
% 
% X = (XX-ones(T,1)*MX)*diag(1./SX);
% Y = (YY-ones(T,1)*MY)*diag(1./SY);
% 
% eval(['save FredMDmediumHor',num2str(hor),' X Y Mnem Time']) 

idLarge  = find(ismember(names,unbal)==0);
Data = yt(StartSample:EndSample,idLarge);
Mnem = names(idLarge);
Tcode = tcode(idLarge);


idTarget  = find(ismember(Mnem,Target));
idTarget2  = find(ismember(Mnem,Target2));
idTarget3  = find(ismember(Mnem,Target3));
YY = filter(ones(1,hor)/hor,1,Data(:,idTarget)); YY = YY(hor+1:end,:);
YY2 = filter(ones(1,hor)/hor,1,Data(:,idTarget2)); YY2 = YY2(hor+1:end,:);
YY3 = filter(ones(1,hor)/hor,1,Data(:,idTarget3)); YY3 = YY3(hor+1:end,:);
XX = Data(1:end-hor,:);

% MY = mean(YY);
% SY =  std(YY);
% 
% MX = mean(XX);
% SX =  std(XX);
% 
% [T,k] = size(XX);
% 
% X = (XX-ones(T,1)*MX)*diag(1./SX);
% Y = (YY-ones(T,1)*MY)*diag(1./SY);
X=XX
Y=YY
Y_inflation=YY2
Y_unrate=YY3
eval(['save FredMDlargeHor',num2str(hor),' X Y Y_inflation Y_unrate Mnem Time']) 


