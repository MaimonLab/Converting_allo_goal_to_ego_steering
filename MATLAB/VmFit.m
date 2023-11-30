clear all
load('vAllData.mat')
cell = VRData(:,4);
HAll = VRData(:,1);
vAll = VRData(:,3);

nCell = 21;
fitData = zeros(nCell,3);
% plot angles
hPlot = -180:1:180;

pVar = zeros(nCell,1);
figure(1)
for i=1:nCell
    HFit = HAll(cell==i);
    vFit = vAll(cell==i);
    amp = 10;
    offSet = -50;
    phase = -90;
    % Do fit
    pFit0 = [amp offSet phase];
    fitData(i,:) = fminsearch(@(pFit) FitCos(pFit,HFit,vFit),pFit0);
    
    % fit parameters
    amp = fitData(i,1);
    offSet = fitData(i,2);
    phase = fitData(i,3);

    % Plot fit results
    figure(1)
    subplot(7,3,i)
    plot(HFit,vFit,'bo','linewidth',2);
    hold on
    plot(hPlot,amp*cos(pi*(hPlot-phase)/180)+offSet,'r','linewidth',2);
    hold off;
    xlim([-180 180])
    if (mod(i,3)==1)
        ylabel('Vm')
    end
    if (i>18)
        xlabel('relative heading')
    end
    title(['cell ' num2str(i)]);
end

function errS = FitCos(pFit,H,v)
    amp = pFit(1);
    offSet = pFit(2);
    phase = pFit(3);
    errS = sum((v-amp*cos(pi*(H-phase)/180)-offSet).^2);
end