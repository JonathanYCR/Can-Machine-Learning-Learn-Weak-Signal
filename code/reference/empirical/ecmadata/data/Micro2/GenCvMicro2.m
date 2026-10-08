close all
clear all

rng(10)

load Data1stStageEminentDomain_u80.mat
data=Data1stStageEminentDomain_u80; clear Data1stStageEminentDomain_u80;
year = unique(Year);
circuit = unique(Circuit);
T = length(year);
N = length(circuit);

%   Name           Size              Bytes  Class     Attributes
% -----------   ----------         -------  ------
%   Circuit      312x1                2496  double   \
%   circuit       12x1                  96  double        
%   Year         312x1                2496  double
%   year          26x1                 208  double 
%   data         312x219            546624  double              
%   l              1x1                   8  double    
%   N              1x1                   8  double              
%   T              1x1                   8  double     

T0est = 6;
Nest = floor(N*.5);
Mcv = 1000;

cont = 0;
for jcv = 1:Mcv
    for jt = T0est:T-1
        cont               =  cont+1;
        CrossPerm          =  randperm(N)';
        CrossEst         =  circuit(CrossPerm(1:Nest));
        CrossEval          =  circuit(CrossPerm(Nest+1:end));
        TimeEst          =  year(1:jt);
        TimeEval           =  year(jt+1);
        Temp               =  (ismember(Year,TimeEst) | ismember(Circuit,CrossEst));
        EstSmplCv{cont}  =  find(Temp==1);
        Temp               =  (ismember(Circuit,CrossEval)  & ismember(Year,TimeEval));
        EvalSmplCv{cont}   =  find(Temp==1);
    end
end;

% save CvSamplesMicro2 EstSmplCv EvalSmplCv