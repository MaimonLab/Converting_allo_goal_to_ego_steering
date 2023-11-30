clear all

% Define PB and FB angles
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

% Choose the goal and heading angle to be plotted
% Note, we are using a coordinate system where 0° is
%   the convention -90° 
% In addition, angles are clockwise positive
G = 0;
H = 45;

% Correct for no-conventional axes
G = G + 180;
H = H + 180;

vML = d*cos(pi*(G-thetaFB)/180)+ ...
           cos(pi*(H-thetaLeft)/180);
vMR = d*cos(pi*(G-thetaFB)/180)+ ...
            cos(pi*(H-thetaRight)/180);
rML = a*log(1+exp(b*(vML+c)));
rMR = a*log(1+exp(b*(vMR+c)));

% Compute turning signal
rLLAL = sum(rML);
rRLAL = sum(rMR);
rTurn = rRLAL-rLLAL;

% This is a vector version of the turning signal for the plots 
% The division by 80 is just to give it a reasonable length
vTurn = rTurn*[1 0]/80;

figure(1)
for i=1:12
    % Plot preference vectors
    subplot(3,12,i)
    quiver(0,0, -sin(pi*thetaFB(i)/180),-cos(pi*thetaFB(i)/180),...
                0,'k','linewidth',2,'MaxHeadSize',2);
    hold on
    quiver(0,0,-sin(pi*thetaRight(i)/180),-cos(pi*thetaRight(i)/180),...
                0,'r','linewidth',2,'MaxHeadSize',2);
    quiver(0,0,-sin(pi*thetaLeft(i)/180),-cos(pi*thetaLeft(i)/180),...
                0,'b','linewidth',2,'MaxHeadSize',2);
    hold off   
    xlim([-1.5 1.5])
    ylim([-1.5 1.5])
    set(gca,'XTick',[])
    set(gca,'YTick',[])     
    subplot(3,12,12+i)
    % Plot left and right PFL3 rates
    bar(1,rMR(i),'r')
    hold on
    bar(2,rML(i),'b')
    hold off
    xlim([0.3 2.7])
    ylim([0 55])
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    % Plot left and right LAL responses
    subplot(3,12,27)
    bar(rLLAL,'b');
    ylim([0 220])
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    subplot(3,12,33)
    bar(rRLAL,'r');
    ylim([0 220])
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    % Plot turning vectors
    subplot(3,12,30)
    quiver(0,0,vTurn(1),vTurn(2),...
                0,'k','linewidth',2,'MaxHeadSize',1);
    xlim([-1.5 1.5])
    ylim([-1.5 1.5])
    set(gca,'XTick',[])
    set(gca,'YTick',[])
end