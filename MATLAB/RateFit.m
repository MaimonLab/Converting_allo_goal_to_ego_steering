clear all
load('rData.mat')

% Heading and goal angles
H = -157.5:45:157.5;
G = H;

% Initial parameter values for fits
a = 25;
b = 1;
c = 0;
d = 1;
phi = 45;

% Fit to Data
pFit0 = [a b c d phi];
OPTIONS = optimset('MaxFunEvals',1e9,'MaxIter',1e6);
pFit = fminsearch(@(pFit) FitErr(pFit,H,G,rate),pFit0,OPTIONS);

% percent variance
pVar = 100*(1-FitErr(pFit,H,G,rate)/(63*var(rate(:))))

% Fit parameters
a = pFit(1);
b = pFit(2);
c = pFit(3);
d = pFit(4);
phi = pFit(5);

% theta is the heading angle variable for plotting
theta = -180:1:180;
rM = zeros(length(G),length(theta));
figure(1)
for i=1:8
    vMa = cos(pi*theta/180)+d*cos(pi*(G(i)-phi)/180);
    rM(i,:) = a*log(1+exp(b*(vMa+c)));
    subplot(2,4,i)
    plot(H,rate(i,:),'bo','linewidth',2)
    hold on
    plot(theta,rM(i,:),'r','linewidth',2)
    hold off
    xlim([-180 180])
    ylim([0 65])
    if (i>4)
        xlabel('relative heading');
    end
    if ((i==1)|(i==5))
        ylabel('firing rate (Hz)');
    end
    title(['rel. goal = ',num2str(G(i)),'°'])
    box off;
end

function errS = FitErr(pFit,H,G,rate)
    errS = 0;
    a = pFit(1);
    b = pFit(2);
    c = pFit(3);
    d = pFit(4);
    phi = pFit(5);
    for i=1:length(G)
        vMa = cos(pi*H/180)+d*cos(pi*(G(i)-phi)/180);
        rM = a*log(1+exp(b*(vMa+c)));
        errS = errS + sum((rM-rate(i,:)).^2);
    end
end