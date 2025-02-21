%% TRAJECTORY PARAMETERS
%Initial conditions
%scaling factor of the trajectory
Ts = 1/60;
eta=0.7;
alpha=2.2;
k=0:Ts:2*pi*alpha*2-Ts;
xr=eta*sin(k/alpha);
yr=eta*sin(k/(2*alpha));

%Velocity trajectory
xpr=eta*cos(k/alpha)*(1/alpha);
ypr=eta*cos(k/(2*alpha))*(1/(2*alpha));

%Acceleration trajectory
xppr=-eta*sin(k/alpha)*(1/alpha)*(1/alpha);
yppr=-eta*sin(k/(2*alpha))*(1/(2*alpha))*(1/(2*alpha));

%Driving velocity reference
vr=sqrt(xpr.^2+ypr.^2);
wr=(yppr.*xpr-xppr.*ypr)./(xpr.^2+ypr.^2);


% Orientation reference
thetar=atan2(ypr,xpr);


vrmax =max(vr)
wrmax = max(wr)

r = 0.04445;
d = 0.393;
wrwlmax = 10.0;


vmax = r*wrwlmax 
wmax = r*wrwlmax/d
