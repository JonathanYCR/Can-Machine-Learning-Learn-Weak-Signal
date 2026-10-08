function f = CVridgeK(theta,y,x,k)

n = size(y,1);
cvpart = cvpartition(n,'Kfold',k);

f = 0;
for rr = 1:k
    trIdx = cvpart.training(rr);
    teIdx = cvpart.test(rr);
    yRTemp = y(trIdx);
    xRTemp = x(trIdx,:);
    bRidge = ridge(yRTemp,xRTemp,theta,0);
    f = f + sum((y(teIdx)-[ones(sum(teIdx),1),x(teIdx,:)]*bRidge).^2);
end
