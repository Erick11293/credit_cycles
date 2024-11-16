%TABLAS LATEX

%Ruta
cd('L:\8_Ciclos_del_crédito');

%Limpiando base
clear;
clc;    
close all;



%Labels de paises
country_n={'CAN','DEU','FRA','UK','ITA','JPN','USA','PER'}
label_param={'\alpha_1',...
             '\alpha_2',...
             '\beta_1',...
             '\beta_2'};

label={'\rho_1^b','\rho_2^b','\rho_1^f','\rho_2^f',...
    '\sigma^2_{gb}','\sigma^2_{cb}','\sigma^2_{gf}','\sigma^2_{cf}',...
    '\sigma^2_{v}', '\kappa', '\zeta_1', '\zeta_2',...
    '\sigma_{\epsilon_f}^2','\sigma^2_{\psi}','p_{11}','p_{22}',...
    '\alpha_1','\alpha_2','\beta_1','\beta_2'};
         
%Paises a reportar
paises=[1,2,3,4,7,8];

%loop

for i_pais=1:length(paises)
    
    data_res=xlsread(fullfile(country_n{paises(i_pais)}, 'res_v2.xlsx'));
    
    c=data_res(17:20,2);
    var_c=data_res(17:20,3);
    
    tabla_latex=[c,c./((var_c).^0.5)];
    
    latextable(tabla_latex,'Horiz',{'Estimates','T'},'Hline',[0,1,NaN],'Vert',label_param,'format','%3.2f','name',fullfile(country_n{paises(i_pais)}, 'table_latex.tex'),'save');
    
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Cuadro de estimaciones finales

tabla_latex=[]

for i_pais=1:length(paises)
    
    data=xlsread(fullfile(country_n{paises(i_pais)}, 'res_v2.xlsx'));
    data_var=xlsread(fullfile(country_n{paises(i_pais)}, 'res_v2.xlsx'),2);
    c=param_finales(data(:,2))';
    g=grad_param(c);
    var_c=diag(g*data_var*g');
    t=c./((var_c).^0.5);
    tabla=[c,c./((var_c).^0.5)];
    
    for j=1:length(c)
        
       tabla_aux(j*2-1,1)= c(j);
       tabla_aux(j*2,1)= t(j);
        
    end
    
    tabla_latex=[tabla_latex, tabla_aux];
    
    
end

tabla_latex=real(tabla_latex);


for j=1:length(c)
    
   label_latex{j*2-1}= label{j};
   label_latex{j*2}= '';
    
    
end
latextable(tabla_latex,'Horiz', country_n(paises),'Hline',[0,1,NaN],'Vert',label_latex,'format','%3.2f','name','table_all.tex','save');




function [c_final]=param_finales(c)

    c_final=[(c(1)/(1+abs(c(1))))+(c(2)/(1+abs(c(2)))),-(c(1)/(1+abs(c(1))))*(c(2)/(1+abs(c(2)))),...
             (c(3)/(1+abs(c(3))))+(c(4)/(1+abs(c(4)))),-(c(3)/(1+abs(c(3))))*(c(4)/(1+abs(c(4)))),...
             c(5)^2, c(6)^2, c(7)^2, c(8)^2,...
             c(9)^2, c(10),(c(11)/(1+abs(c(11))))+(c(12)/(1+abs(c(12)))),-(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12)))),...
             c(13)^2, c(14)^2, exp(0)/(exp(0)+exp(c(15))), exp(0)/(exp(0)+exp(c(16))),...
             c(17),c(18), c(19), c(20)];

end


function g=grad_param(c)

    c_back=c;
    
    c_final_back=param_finales(c);
    
    for i=1:length(c)

        c=c_back;
        c(i)=c(i)+0.0001;
        c_final=param_finales(c);
        g(:,i)=(c_final-c_final_back)/0.0001;

    end



end