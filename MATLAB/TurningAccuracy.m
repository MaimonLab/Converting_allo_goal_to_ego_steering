clear all

thetaRight = [67.5 112.5 157.5 157.5 -157.5 -112.5 -112.5 ...
                -67.5 -22.5 -22.5 22.5 67.5];
thetaLeft = [-67.5 -22.5 22.5 22.5 67.5 112.5 112.5 ...
                157.5 -157.5 -157.5 -112.5 -67.5];
thetaFB = (0:30:330) + 15;

% Parameters from data fit
a = 29.2282;
b = 2.1736;
c = -0.7011;
d = 0.6299;

% goal angles
G = -180:1:180;
G(end) = [];

turnEr = zeros(length(G),1);
for i=1:length(G)
    HZero = fzero(@(H) turnFunc(H,G(i),thetaLeft,thetaRight,...
        thetaFB,a,b,c,d),G(i));
    turnErr(i) = HZero-G(i);
end
sqrt(mean(turnErr.^2))

figure(1)
plot(G,turnErr,'b','linewidt',2)
xlim([-180 180])
xlabel('goal angle (deg)')
ylabel('heading error (deg)')

function turnS = turnFunc(H,G,thetaLeft,thetaRight,thetaFB,a,b,c,d)
    vML = d*cos(pi*(G+thetaFB)/180)+ ...
       cos(pi*(H+thetaLeft)/180);
    vMR = d*cos(pi*(G+thetaFB)/180)+ ...
        cos(pi*(H+thetaRight)/180);
    turnS = sum(a*log(1+exp(b*(vMR'+c)))-a*log(1+exp(b*(vML'+c))));
end

