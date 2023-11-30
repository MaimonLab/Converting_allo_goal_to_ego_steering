clear all

thetaRight = -[67.5 112.5 157.5 157.5 -157.5 -112.5 -112.5 ...
                -67.5 -22.5 -22.5 22.5 67.5];
thetaLeft = -[-67.5 -22.5 22.5 22.5 67.5 112.5 112.5 ...
                157.5 -157.5 -157.5 -112.5 -67.5];
thetaFB = -((0:30:330) + 15);

% Parameters from data fit
a = 29.2282;
b = 2.1736;
c = -0.7011;
d = 0.6299;

% heading and goal angles
% iGZero is the index of thetaGH = 0
H = -180:1:180;
G = H;
iGZero = find(G==0);

% Left and Right PFL3 rates
%   indexed by (heading,goal,thetaFB)
rML = zeros(length(H),length(G),12);
rMR = zeros(length(H),length(G),12);

for i=1:12
    for j=1:length(G)
        vML = d*cos(pi*(G(j)-thetaFB(i))/180)+ ...
           cos(pi*(H-thetaLeft(i))/180);
        vMR = d*cos(pi*(G(j)-thetaFB(i))/180)+ ...
            cos(pi*(H-thetaRight(i))/180);
        rML(:,j,i) = a*log(1+exp(b*(vML+c)));
        rMR(:,j,i) = a*log(1+exp(b*(vMR+c)));
    end
end

% Left and right LAL outputs and turning signal
rLLAL = sum(rML,3);
rRLAL = sum(rMR,3);
rTurn = rRLAL-rLLAL;

LeftLAL = rLLAL(:,iGZero)-mean(rLLAL(:,iGZero));
RightLAL = rRLAL(:,iGZero)-mean(rRLAL(:,iGZero));
turn = rTurn(:,iGZero);

% Plot LAL and turning signals for goal angle = 0
% The mean of the LAL signals is subtracted out here
figure(1)
subplot(3,1,1)
plot(H,LeftLAL,'b','linewidth',2)
hold on
plot([-180 180],[0 0],'k','linewidth',1);
plot([0 0],[-100 100],'k','linewidth',1);
hold off;
xlim([-180 180]);
set(gca,'YTickLabel',[]);
title('left LAL')
box off
subplot(3,1,2)
plot(H, RightLAL,'r','linewidth',2)
hold on
plot([-180 180],[0 0],'k','linewidth',1);
plot([0 0],[-100 100],'k','linewidth',1);
hold off;
xlim([-180 180]);
set(gca,'YTickLabel',[]);
title('right LAL')
box off
subplot(3,1,3)
plot(H,turn,'k','linewidth',2)
hold on
plot([-180 180],[0 0],'k','linewidth',1);
plot([0 0],[-180 180],'k','linewidth',1);
hold off;
xlim([-180 180]);
ylim([-180 180]);
xlabel('relative heading angle')
set(gca,'YTickLabel',[]);
title('turning signal')
box off

