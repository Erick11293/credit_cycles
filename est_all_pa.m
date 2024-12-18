%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PROGRAMA PARA CORRER FILTRO PARA PAISES DEL G7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Ruta
cd('G:\8_Ciclos_del_cr�dito');

%Limpiando base
clear;
clc;    
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Labels to non observable states
labels_no_states={'Ouput Trend','Ouput Trend Growth','Output cycle','Credit Trend','Credit Trend Growth','Credit cycle','Output cycle innovation','Credit cycle innovation','Long run inflation'};

%Non observable states
no_states=[1,2,3,5,6,7,9];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Labels de paises
country_n={'CAN','DEU','FRA','UK','ITA','JPN','USA','PER'}
    
%Paises a estimar
paises=[7];
% paises=[1];

estimacion=[1,0];

%Opciones de estimacion
tol=0.00001;
n_eval=2000;
n_eval_p=1000000;
tol2=0.00001;


%Creando trimestres 

years = 1970:2017;
numyears = length(years); 
jan1s = zeros(numyears,1); 
quarters = zeros(4*numyears,1);

for Yearno = 1:numyears

    thisyear = years(Yearno); 
    jan1s(Yearno) = datenum([thisyear 1 1]); 
    quarters((Yearno-1)*4+1) = jan1s(Yearno); 
    quarters((Yearno-1)*4+2) = datenum([thisyear 4 1]); 
    quarters((Yearno-1)*4+3) = datenum([thisyear 7 1]); 
    quarters((Yearno-1)*4+4) = datenum([thisyear 10 1]); 

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Cargando data

%GDP
[data_gdp,fecha]=xlsread('bd_cred_all.xlsx','bd_gdp');
%CPI
[data_cpi,fecha]=xlsread('bd_cred_all.xlsx','bd_cpi');
%CRED_REAL
[data_cred,fecha]=xlsread('bd_cred_all.xlsx','bd_cred_real');

%PRECIOS DE ACTIVOS
[data_pa,fecha]=xlsread('bd_cred_all.xlsx','bd_pa');


%Valores iniciales
[data_vi,fecha]=xlsread('bd_cred_all.xlsx','vi');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Modelos

%Modelo 1
%Sin interdependencia - ni MS

mod1_str=["loglkalmanms(obs," ...
        "cat(3,s_p)," ...
        "cat(3,P_p)," ...
        "cat(3,[0,0,0;0,0,0;0,(c(11)/(1+abs(c(11))))+(c(12)/(1+abs(c(12)))),-(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0]),"...
        "[x]," ...
        "cat(3,[1,0,1,0,0,0,0,0,0;0,0,0,0,1,0,1,0,0;0,0,abs(c(10)),0,0,0,0,0,1-(c(11)/(1+abs(c(11))))-(c(12)/(1+abs(c(12))))+(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0,0,0,0,abs(c(17)),0,1]),"...
        "cat(3,diag([(c(15))^2,(c(9))^2,(c(13))^2,c(16)^2]))," ...
        "cat(3,[1,1,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0,0;0,0,(c(1)/(1+abs(c(1))))+(c(2)/(1+abs(c(2)))),-(c(1)/(1+abs(c(1))))*(c(2)/(1+abs(c(2)))),0,0,0,0,0;0,0,1,0,0,0,0,0,0;0,1,0,0,1,1,0,0,0,;0,0,0,0,0,1,0,0,0;0,0,0,0,0,0,(c(3)/(1+abs(c(3))))+(c(4)/(1+abs(c(4)))),-(c(3)/(1+abs(c(3))))*(c(4)/(1+abs(c(4)))),0;0,0,0,0,0,0,1,0,0;0,0,0,0,0,0,0,0,1])," ...
        "cat(3,[0;0;0;0;0;0;0;0;0])," ...
        "cat(3,diag([0,c(5)^2,(c(6))^2,0,0,c(7)^2,(c(8))^2,0,c(14)^2])),"...
        "[1],"...
        "[1], "];



%Modelo 2
%Varianzas
var_aux=zeros(9,9);
var_aux(3,7)=1;
var_aux(7,3)=1;

%Incluyendo interdependencia - con MS
mod2_str=["loglkalmanms(obs," ...
        "cat(3,s_p,s_p,s_p)," ...
        "cat(3,P_p,P_p,P_p)," ...
        "cat(3,[0,0,0;0,0,0;0,(c(11)/(1+abs(c(11))))+(c(12)/(1+abs(c(12)))),-(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0],"...
            "[0,0,0;0,0,0;0,(c(11)/(1+abs(c(11))))+(c(12)/(1+abs(c(12)))),-(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0],"...
            "[0,0,0;0,0,0;0,(c(11)/(1+abs(c(11))))+(c(12)/(1+abs(c(12)))),-(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0]),"...
        "[x]," ...
        "cat(3,[1,0,1,0,0,0,0,0,0;0,0,0,0,1,0,1,0,0;0,0,abs(c(10)),0,0,0,0,0,1-(c(11)/(1+abs(c(11))))-(c(12)/(1+abs(c(12))))+(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0,0,0,0,abs(c(27)),0,1],"...
              "[1,0,1,0,0,0,0,0,0;0,0,0,0,1,0,1,0,0;0,0,abs(c(10)),0,0,0,0,0,1-(c(11)/(1+abs(c(11))))-(c(12)/(1+abs(c(12))))+(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0,0,0,0,abs(c(27)),0,1]," ...
              "[1,0,1,0,0,0,0,0,0;0,0,0,0,1,0,1,0,0;0,0,abs(c(10)),0,0,0,0,0,1-(c(11)/(1+abs(c(11))))-(c(12)/(1+abs(c(12))))+(c(11)/(1+abs(c(11))))*(c(12)/(1+abs(c(12))));0,0,0,0,0,0,abs(c(27)),0,1])," ...
        "cat(3,diag([(c(15))^2,(c(9))^2,(c(13))^2,c(26)^2]),diag([(c(15))^2,(c(9))^2,(c(13))^2,c(26)^2]),diag([(c(15))^2,(c(9))^2,(c(13))^2,c(26)^2]))," ...
        "cat(3,[1,1,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0,0;0,0,(c(1)/(1+abs(c(1))))+(c(2)/(1+abs(c(2)))),-(c(1)/(1+abs(c(1))))*(c(2)/(1+abs(c(2)))),0,0,0,0,0;0,0,1,0,0,0,0,0,0;0,1,0,0,1,1,0,0,0,;0,0,0,0,0,1,0,0,0;0,0,0,0,0,0,(c(3)/(1+abs(c(3))))+(c(4)/(1+abs(c(4)))),-(c(3)/(1+abs(c(3))))*(c(4)/(1+abs(c(4)))),0;0,0,0,0,0,0,1,0,0;0,0,0,0,0,0,0,0,1]," ...
              "[1,1,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0,0;0,0,(c(1)/(1+abs(c(1))))+(c(2)/(1+abs(c(2)))),-(c(1)/(1+abs(c(1))))*(c(2)/(1+abs(c(2)))),0,0,0,0,0;0,0,1,0,0,0,0,0,0;0,1,0,0,1,1,0,0,0,;0,0,0,0,0,1,0,0,0;0,0,0,0,0,0,(c(3)/(1+abs(c(3))))+(c(4)/(1+abs(c(4)))),-(c(3)/(1+abs(c(3))))*(c(4)/(1+abs(c(4)))),0;0,0,0,0,0,0,1,0,0;0,0,0,0,0,0,0,0,1]," ...
              "[1,1,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0,0;0,0,(c(1)/(1+abs(c(1))))+(c(2)/(1+abs(c(2)))),-(c(1)/(1+abs(c(1))))*(c(2)/(1+abs(c(2)))),0,0,0,0,0;0,0,1,0,0,0,0,0,0;0,1,0,0,1,1,0,0,0,;0,0,0,0,0,1,0,0,0;0,0,0,0,0,0,(c(3)/(1+abs(c(3))))+(c(4)/(1+abs(c(4)))),-(c(3)/(1+abs(c(3))))*(c(4)/(1+abs(c(4)))),0;0,0,0,0,0,0,1,0,0;0,0,0,0,0,0,0,0,1])," ...
        "cat(3,[0;0;0;0;0;0;0;0;0],[0;0;0;0;0;0;0;0;0],[0;0;0;0;0;0;0;0;0])," ...
        "cat(3,diag([0,c(5)^2,(c(6))^2,0,0,c(7)^2,(c(8))^2,0,0]),"...
            "diag([0,c(5)^2,((c(6))^2)*(c(22)^2+1)^2,0,0,c(7)^2,((c(8))^2),0,c(14)^2]),"...
            "diag([0,c(5)^2,((c(6))^2)+c(23)^2*((c(8))^2),0,0,c(7)^2,((c(8))^2)+c(24)^2*(c(6))^2,0,c(25)^2]) + var_aux * (abs(c(23))*((c(8))^2)+abs(c(24))*((c(6))^2)))," ...
        "[exp(0)/(exp(0)+exp(c(16))+exp(c(17))), exp(c(18))/(exp(c(18))+exp(0)+exp(c(19))), exp(c(20))/(exp(c(20))+exp(c(21))+exp(0));"...
            "exp(c(16))/(exp(0)+exp(c(16))+exp(c(17))), exp(0)/(exp(c(18))+exp(0)+exp(c(19))), exp(c(21))/(exp(c(20))+exp(c(21))+exp(0));"...
            "exp(c(17))/(exp(0)+exp(c(16))+exp(c(17))), exp(c(19))/(exp(c(18))+exp(0)+exp(c(19))), exp(0)/(exp(c(20))+exp(c(21))+exp(0))],"...
        "[0.30;0.30;0.4], "];
    

    
%Resto de opciones
    
mod_est=["false, false)"];
mod_suav=["true, false)"];

mod1_est=strjoin([mod1_str mod_est]);
mod1_suav=strjoin([mod1_str mod_suav]);    

mod2_est=strjoin([mod2_str mod_est]);
mod2_suav=strjoin([mod2_str mod_suav]);

%Creando estimaciones
estim1="c=fminsearch (@(c) -"+mod1_est+",vi,optimset('MaxFunEval',n_eval,'MaxIter',n_eval,'Display','iter','PlotFcns',@optimplotx,'TolFun',tol,'TolX',tol));";
estim2="c=fminsearch (@(c) -"+mod2_est+",vi,optimset('MaxFunEval',n_eval,'MaxIter',n_eval,'Display','iter','PlotFcns',@optimplotx,'TolFun',tol,'TolX',tol));";


%Creando opciones de estimacion
options=optimoptions('fminunc','MaxIterations',n_eval_p,'MaxFunctionEvaluations',n_eval_p,'Display','iter','PlotFcns',@optimplotx);

%Creando estimaciones
estim1_p="c=fminunc (@(c) -"+mod1_est+",vi,options);";
estim2_p="c=fminunc (@(c) -"+mod2_est+",vi,options);";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Loop para generar estimaciones y graficos por pa�s

for i_pais=1:length(paises)
    
    %Creando carpeta
    mkdir(country_n{paises(i_pais)});
    
    %Determinando fecha de inicio
    f_1=sum(isnan(data_gdp(:,paises(i_pais))));
    f_2=sum(isnan(data_cpi(:,paises(i_pais))));
    f_3=sum(isnan(data_cred(:,paises(i_pais))));
    f_4=sum(isnan(data_pa(:,paises(i_pais))));
    fecha_ini=max(max(max(f_1,f_2),f_3),f_4)+7;
    
    %Obteniendo data
    %Logaritmo del pbi
    l_y=log(data_gdp(fecha_ini:end,paises(i_pais)))*100;

    %Logaritmo del credito
    l_cred=log(data_cred(fecha_ini:end,paises(i_pais)))*100;
   
    %Inflacion
    inf=(log(data_cpi(fecha_ini:end,paises(i_pais)))-log(data_cpi(fecha_ini-1:end-1,paises(i_pais))))*100;
    inf_1=(log(data_cpi(fecha_ini-1:end-1,paises(i_pais)))-log(data_cpi(fecha_ini-2:end-2,paises(i_pais))))*100;
    inf_2=(log(data_cpi(fecha_ini-2:end-2,paises(i_pais)))-log(data_cpi(fecha_ini-3:end-3,paises(i_pais))))*100;
    
    %Retornos de activos financieros
    
    ret=(log(data_pa(fecha_ini:end,paises(i_pais)))-log(data_pa(fecha_ini-1:end-1,paises(i_pais))))*100;
    
    %Matriz de unos
    x0=ones(length(l_y),1);
    
    %Generando inputs para los filtros
    obs=[l_y,l_cred,inf,ret];
    x=[x0,inf_1,inf_2];
    
    %Matrices de medias iniciales
    %Modelo 1 y 2
    s_p=[l_y(1);mean(l_y(2:end)-l_y(1:end-1));0;0;l_cred(1);mean(l_cred(2:end)-l_cred(1:end-1)-(l_y(2:end)-l_y(1:end-1)));0;0;mean(inf)];
    
    %Matrices de varianzas iniciales
    %Modelo 1 y 2
    P_p=diag([(l_y(1)/500)^2;0.25;0.25;0.25;(l_cred(1)/500)^2;0.25;0.25;0.25;0.25]);
    
    %Definiendo valores iniciales para el modelo 1
    c(data_vi(isnan(data_vi(:,9))~=1,9),1)=data_vi(isnan(data_vi(:,9))~=1,paises(i_pais));
    
%     c=data_vi(1:15,paises(i_pais));
    vi=c;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Filtro HP
    
    [trend,cycle]=hpfilter(l_y,1600);
    [trend_cred,cycle_cred]=hpfilter(l_cred,1600);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %MODELO 1
    
    %Estimacion mediante Newton rhapson
    if estimacion(1)==1
 
        error_conv=1000;
        while error_conv>tol2
            vi=c;
            eval(estim1_p);
%             vi=c;
%             eval(estim1);
            error_conv=norm(vi-c);
        end
        
    end
    
    %Estimacion que converge
    
    
    %Guardando resultados
    
    c1_estim(:,paises(i_pais))=c;
    
    %Resultados
    
    [LL,S_pred,V_pred,Pr_pred,S2,V2,Pr,S1,V1,Pf,SS,l_obs]=eval(mod1_suav);
    
    %Hessiano y gradiente
    [h_LL,d_LL]=hessian_log_ml(c,mod1_est,obs,s_p,P_p,x);
    
    %Varianzsa del modelo 1
    var_c=inv(-h_LL);
    
    %Tabla de resultados
    Resultados=table([1:length(c)]',c,diag(var_c),c./(diag(var_c).^0.5),d_LL',diag(h_LL));
    Resultados.Properties.VariableNames = {'Param','Estimates','Variances','T','Gradient','Hessian'}
    
    filename = country_n{paises(i_pais)}+"\res_v1.xlsx"
    writetable(Resultados,filename,'Sheet',1,'Range','D1')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Graficos
    
    figure;
    plot(quarters(fecha_ini:end),SS(1,:),quarters(fecha_ini:end),trend);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Potenciales Filtro vs HP')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '11_pot'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(2,:));
    xlim([quarters(fecha_ini) quarters(end-1)]);
    datetick('x','yyyy','keeplimits');
%     title('Crecimiento potencial Filtro');
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '12_growth_pot'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(3,:),quarters(fecha_ini:end),cycle);
    xlim([quarters(fecha_ini) quarters(end-1)]);
    datetick('x','yyyy','keeplimits');
%     title('Brecha Filtro vs HP');
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '13_brecha'), 'jpeg')
    
    figure;
    subplot(2,1,1), plot(quarters(fecha_ini:end),SS(9,:),quarters(fecha_ini:end),inf);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
        
    subplot(2,1,2), plot(quarters(fecha_ini+3:end),((1+SS(9,1:end-3)/100).*(1+SS(9,2:end-2)/100).*(1+SS(9,3:end-1)/100).*(1+SS(9,4:end)/100)-1)*100,quarters(fecha_ini+3:end),(((1+inf(1:end-3,1)/100).*(1+inf(2:end-2,1)/100).*(1+inf(3:end-1,1)/100).*(1+inf(4:end,1)/100))-1)*100);
    xlim([quarters(fecha_ini+3) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Inflacion esperada')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '14_inflacion'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(5,:),quarters(fecha_ini:end),trend_cred);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Potenciales Filtro vs HP')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '15_pot_cred'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(5,:),quarters(fecha_ini:end),l_cred);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Potenciales Filtro vs Serie')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '16_pot_cred_v2'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(6,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Crecimiento potencial Filtro')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '17_growth_pot_cred'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(7,:),quarters(fecha_ini:end),cycle_cred);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Brecha Filtro vs HP')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '18_brecha_cred'), 'jpeg')
    
    
    close all;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %MODELO 2 
    
    %Valores iniciales de parametros faltantes
    c(data_vi(isnan(data_vi(:,10))~=1,10),1)=data_vi(isnan(data_vi(:,10))~=1,paises(i_pais));
    vi=c;
    
    %Estimacion Newton Rhapson
    
    if estimacion(2)==1
        error_conv=1000;
        
        while error_conv>tol2
            vi=c;
            eval(estim2_p);
            error_conv=max(abs(vi-c));
        end
    end
    
    %Estimacion
    
    
    %Guardando estimaciones
    
    c2_estim(:,paises(i_pais))=c;
    
    %Resultados
    
    [LL,S_pred,V_pred,Pr_pred,S2,V2,Pr,S1,V1,Pf,SS,l_obs]=eval(mod2_suav);
    
    %Hessiano y gradiente
    [h_LL,d_LL]=hessian_log_ml(c,mod2_suav,obs,s_p,P_p,x);
    
    %Varianzsa del modelo 1
    var_c=inv(-h_LL);
    
    %Tabla de resultados
    Resultados=table([1:length(c)]',c,diag(var_c),c./(diag(var_c).^0.5),d_LL',diag(h_LL));
    Resultados.Properties.VariableNames = {'Param','Estimates','Variances','T','Gradient','Hessian'}
    
    var_res=table(var_c);
    
    filename = country_n{paises(i_pais)}+"\res_v2.xlsx"
    writetable(Resultados,filename,'Sheet',1,'Range','D1')
    writetable(var_res,filename,'Sheet',2,'Range','A1')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Fechas de sombreados
    
    %Generando fechas de sombreado
    
    fechas_state2=[Pf(2,:)>0.9];
    fechas_somb=[];
    i_somb=1;
    
    if fechas_state2(1)==1
        fechas_somb(i_somb,1)=[1];
        fechas_somb(i_somb,4)=[1];
    end
    
    for i_fechas_state2=2:length(fechas_state2)
        if fechas_state2(i_fechas_state2-1)==1 & fechas_state2(i_fechas_state2)==0
            fechas_somb(i_somb,2)=i_fechas_state2;
            fechas_somb(i_somb,3)=i_fechas_state2;
            i_somb=i_somb+1
        end
        
        if fechas_state2(i_fechas_state2-1)==0 & fechas_state2(i_fechas_state2)==1
            fechas_somb(i_somb,1)=i_fechas_state2;
            fechas_somb(i_somb,4)=i_fechas_state2;
        end
    end
    
    if size(fechas_somb,1)==i_somb

        if fechas_somb(i_somb,2)==0

            fechas_somb(i_somb,2)=length(fechas_state2);
            fechas_somb(i_somb,3)=length(fechas_state2);

        end


    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Gr�ficos
    
    figure;
    subplot(3,1,1), plot(quarters(fecha_ini:end),Pf(1,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    subplot(3,1,2), plot(quarters(fecha_ini:end),Pf(2,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    
    subplot(3,1,3), plot(quarters(fecha_ini:end),Pf(3,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    
    
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '201_pr_est_s'), 'jpeg')

    figure;
    subplot(3,1,1), plot(quarters(fecha_ini:end),Pr(1,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    subplot(3,1,2), plot(quarters(fecha_ini:end),Pr(2,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    subplot(3,1,3), plot(quarters(fecha_ini:end),Pr(3,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    
        
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '201_pr_est_f'), 'jpeg')

    
    figure;
    plot(quarters(fecha_ini:end),SS(1,:),quarters(fecha_ini:end),trend);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Potenciales Filtro vs HP')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '21_pot'), 'jpeg')
   
    figure;
    plot(quarters(fecha_ini:end),SS(2,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Crecimiento potencial Filtro')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '22_growth_pot'), 'jpeg')

    figure;
    plot(quarters(fecha_ini:end),SS(3,:),quarters(fecha_ini:end),cycle);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Brecha Filtro vs HP')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '23_brecha'), 'jpeg')

    figure;
    subplot(2,1,1), plot(quarters(fecha_ini:end),SS(9,:),quarters(fecha_ini:end),inf);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    
    subplot(2,1,2), plot(quarters(fecha_ini+3:end),((1+SS(9,1:end-3)/100).*(1+SS(9,2:end-2)/100).*(1+SS(9,3:end-1)/100).*(1+SS(9,4:end)/100)-1)*100,quarters(fecha_ini+3:end),(((1+inf(1:end-3,1)/100).*(1+inf(2:end-2,1)/100).*(1+inf(3:end-1,1)/100).*(1+inf(4:end,1)/100))-1)*100);
    xlim([quarters(fecha_ini+3) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%   
%     title('Inflacion esperada')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '24_inflacion'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(5,:),quarters(fecha_ini:end),trend_cred);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Potenciales Filtro vs HP')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '25_pot_cred'), 'jpeg')

    figure;
    plot(quarters(fecha_ini:end),SS(5,:),quarters(fecha_ini:end),l_cred);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Potenciales Filtro vs Serie')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '26_pot_cred_v2'), 'jpeg')

    figure;
    plot(quarters(fecha_ini:end),SS(6,:));
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
%     title('Crecimiento potencial Filtro')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '27_growth_pot_cred'), 'jpeg')
    
    figure;
    plot(quarters(fecha_ini:end),SS(7,:),quarters(fecha_ini:end),cycle_cred);
    axes=gca;
    yaxes=axes.YLim;

    figure;
%     xlim([quarters(fecha_ini) quarters(end-1)])
%     datetick('x','yyyy','keeplimits')
    
        
    hold on;
    for i_somb=1:size(fechas_somb,1)
        patch(quarters(fechas_somb(i_somb,:)+fecha_ini-1), [yaxes(1) yaxes(1),yaxes(2) yaxes(2)], [0.8 0.8 0.8],'EdgeColor','none');
    end    
    plot(quarters(fecha_ini:end),SS(7,:),quarters(fecha_ini:end),cycle_cred);
    xlim([quarters(fecha_ini) quarters(end-1)])
    datetick('x','yyyy','keeplimits')
    hold off;
    
%     title('Brecha Filtro vs HP')
    saveas(gcf,fullfile(country_n{paises(i_pais)}, '28_brecha_cred'), 'jpeg')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Grafico de anexos
    
    
    figure;
    
    for i_graph=1:length(no_states)
        
        serie=SS(no_states(i_graph),:);

        subplot(3,3,i_graph);
        graph_sp=plot(quarters(fecha_ini:end),serie);
        axes=gca;
        yaxes=axes.YLim;
        
        hold on;
        for i_somb=1:size(fechas_somb,1)
            patch(quarters(fechas_somb(i_somb,:)+fecha_ini-1), [yaxes(1) yaxes(1),yaxes(2) yaxes(2)], [0.8 0.8 0.8],'EdgeColor','none');
        end
        plot(quarters(fecha_ini:end),serie);
        title(['\fontsize{8}' labels_no_states{i_graph}]) 
        xlim([quarters(fecha_ini) quarters(end-1)])
        datetick('x','yyyy','keeplimits')
        ytickformat('%.1f')
        set(gca, 'FontSize', 6)
        hold off;
    
    end
    
    saveas(gcf,fullfile(country_n{paises(i_pais)}, 'appendix'), 'jpeg')
    
    
    
    close all;
    
end


save('Param_finales.mat','c1_estim');
save('Param_finales.mat','c2_estim','-append');