clear all
load('RvsVData')

G = [-157.5:45:157.5]';

v = reshape(RvsV(:,1),10,8)';
r = reshape(RvsV(:,2),10,8)';
nCase = zeros(8,1);
for i=1:8
    nCase(i) = sum(~isnan(r(i,:)));
end
vDat = reshape(v',80,1);
rDat = reshape(r',80,1);
iOut = find(isnan(rDat));
vDat(iOut) =[];
rDat(iOut) = [];

sym = ['bo';'b+';'bs';'bx';'ro'; 'r+';'rs';'rx'];

alpha = 7;
beta = 0.1;
shift = 60*ones(1,8);
pFit0 = [alpha beta shift];
OPTIONS = optimset('MaxFunEvals',1e9,'MaxIter',1e6);
pFit = fminsearch(@(pFit) FitErr(pFit,vDat,rDat,nCase),pFit0,OPTIONS);

alpha = pFit(1);
beta = pFit(2);
shift = pFit(3:end);

vPlot = -30:1:20;
rPlot = alpha*log(1+exp(beta*(vPlot)));

pFit0 = [58 8 -60];
pFitSine = fminsearch(@(pFit) FitSine(pFit,G,shift'),pFit0,OPTIONS);

figure(1)
for i=1:8
    plot(v(i,:),r(i,:),sym(i,:),'linewidth',2)
     hold on
end
hold off;

figure(2)
for i=1:8
    plot(v(i,:)+shift(i),r(i,:),sym(i,:),'linewidth',2)
    hold on
    if (i==1)
        plot(vPlot,rPlot,'k','linewidth',2)
    end 
end
hold off;
box off
xlabel('shifted Vm')
ylabel('firing rate (Hz)')

theta = -180:1:180;
figure(3)
plot(G,shift,'bo','linewidth',2)
hold on;
plot(theta,pFitSine(1)*cos(pi*(theta-pFitSine(3))/180)+pFitSine(2), ...
    'b','linewidth',2);
hold off;
xlim([-180 180]);

function errR = FitErr(pFit,vDat,rDat,nCase)
    errR = 0;
    a = pFit(1);
    b = pFit(2);
    s = zeros(size(rDat));
    j = 1;
    for i=1:8
        s(j:j+nCase(i)-1) = pFit(2+i);
        j = j+nCase(i);
    end
    rM = a*log(1+exp(b*(vDat+s)));
    errR = sum((rM-rDat).^2);
end

function errSFit = FitSine(pFit,G,dat)
    fit = pFit(1)*cos(pi*(G-pFit(3))/180) + pFit(2);
    errSFit = sum((fit-dat).^2);
end

