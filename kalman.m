function [s_f,P_f,y_p,var_p]=kalman(B,x,A,T,P_p,s_p,y)
% Script para estimar el modelo de Lanbauch y William (2002)

%Replicando filtro de kalman

%Represntación estado espacio del modelo
%y=As+Bx+e
%s=mu+Rs(-1)+v

%E(e'e)=T
%E(v'v)=K

%Vector de señal (t-1)

y_p=A*s_p+B*x;

%Varianza de la proyeccion

var_p=A*P_p*A'+T;

%Filtro de estado

s_f=s_p+P_p*A'*(A*P_p*A'+T)^(-1)*(y-y_p);

%Filtro de covarianzas

P_f=P_p-P_p*A'*(A*P_p*A'+T)^(-1)*A*P_p;
