function [Ms_sample_ext,M_pred,P_pred,Mr,Vr,Ms,Vs]=loglkalmanms_bayesian(y_r,s_p,P_p,B,x,A,T,R,mu,K,Pr,Pr0,suav,vinic)



%Represntación estado espacio del modelo
%y=As+Bx+e
%s=mu+Rs(-1)+v

%E(e'e)=T
%E(v'v)=K
%las matirces A, B, R, T, K, s_p, P_p edben ser de tres dimensiones. LA
%tercera dimension corresponde a la variabilidad de parametros del MS. La
%matriz Pr debe de ser de nxn donde n es el número de estados del MS
%Pr0 es la matriz de probabilidades inicial

%Solo admite matrices T y K diagonales.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=length(Pr);
N=length(y_r);
n_no=size(R,2);
n_o=size(y_r,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sampling condicional
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%MEJORAR
s_sample(:)=zeros(N,1)+1;

if m>1
    
    for i=2:m
        
        s_sample(:)=s_sample(:)+[rand(N,1)>0.5];
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Identificando estados sin informacion

%No observables
est_inf=[];

for i_si=1:n_no
    
    if K(i_si,i_si)~=0
        
        est_inf=[est_inf,i_si];
    end
end


est_si=setdiff([1:n_no],est_inf);

%Observables

est_obs_inf=[];

for i_si=1:n_o
    
    if T(i_si,i_si)~=0
        
        est_obs_inf=[est_obs_inf,i_si];
    end
end


est_obs_si=setdiff([1:n_o],est_obs_inf);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Estados con informacion
n_no_inf=length(est_inf);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Reservando espacio en memoria


M_pred=zeros(n_no,N);
P_pred=zeros(n_no,n_no,N);

y_p=zeros(n_o,N);
var_p=zeros(n_o,n_o,N);

Mr=zeros(n_no,N);
Vr=zeros(n_no,n_no,N);

Ms=zeros(n_no,N);
Vs=zeros(n_no,n_no,N);

%Probabilidades predecidas de estados

Pr_pred=zeros(m,N);

%Probabilidades filtradas de estados

Pr_filt=zeros(m,N);

%Probabilidades suavizadas de estados | muestreo

Pr_suav=zeros(m,N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Medias incondicionales

if vinic==true
    
    for h=1:length(Pr)
        
        %Medias incondicionales de las variables estados para cada estado MS
        s_p(:,:,h)=((eye(size(R,1))-R(:,:,h))^(-1))*mu(:,:,h);
        
        %Varianzas incondicionales
        K_aux=K(:,:,h);
        
        P_p(:,:,h)=reshape(((eye(size(R,1)^2)-kron(R(:,:,h),R(:,:,h)))^(-1))*K_aux(:),size(R,1),size(R,1));
        
    end
    
    [aux2,aux1]=eig(Pr);
    
    [aux1_1,aux1_2]=sort(diag(aux1));
    
    Pr0=aux2(:,aux1_2(logical([aux1_1>0.99]+[aux1_1<1.01])));
    
    Pr0=Pr0/sum(Pr0);
    
    clear aux1 aux2 aux1_1 aux1_2 K_aux
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MUESTRO DE NO OBSERVABLES



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FILTRO DE KALMAN

for i=1:length(y_r)
    
    %Predecir medias condicionalmente a los estados
    
    if i==1
        
        M_pred(:,i)=(R(:,:,s_sample(i))*s_p(:,:,s_sample(i))+mu(:,:,s_sample(i)));
        P_pred(:,:,i)=R(:,:,s_sample(i))*P_p(:,:,s_sample(i))*R(:,:,s_sample(i))'+K(:,:,s_sample(i));
        
    else
        
        M_pred(:,i)=(R(:,:,s_sample(i))*Mr(:,i-1)+mu(:,:,s_sample(i)));
        P_pred(:,:,i)=R(:,:,s_sample(i))*Vr(:,:,i-1)*R(:,:,s_sample(i))'+K(:,:,s_sample(i));
        
    end
    
    %Filtro de kalman
    
    [Mr(:,i),Vr(:,:,i),y_p(:,i),var_p(:,:,i)]=kalman(B(:,:,s_sample(i)),x(i,:)',A(:,:,s_sample(i)),T(:,:,s_sample(i)),P_pred(:,:,i),M_pred(:,i),y_r(i,:)');
    
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MUESTREO CARTER KOHN


%Ultimo periodo


Ms(:,N)=Mr(:,N);
Vs(:,:,N)=Vr(:,:,N);
Vs(:,:,N)=(Vs(:,:,N)+Vs(:,:,N)')/2;

Ms_sample_ext(:,N)=mvnrnd(Ms(:,N),Vs(:,:,N));
Ms_sample(:,N)=Ms_sample_ext(est_inf,N);

for i=1:N-1
    
    %Matriz de actualización
    J_aux=Vr(:,:,N-i)*R(est_inf,:,s_sample(i))'*(P_pred(est_inf,est_inf,N-i+1)^(-1));
    
    %Media de los estados no observables
    
    Ms(:,N-i)=Mr(:,N-i)+J_aux*(Ms_sample(:,N-i+1)-M_pred(est_inf,N-i+1));
    
    Vs(:,:,N-i)=Vr(:,:,N-i)-J_aux*(P_pred(est_inf,est_inf,N-i+1))*J_aux';
    
    %Forzando simetria (problema de inexactitud de los calculos)
    
    Vs(:,:,N-i)=(Vs(:,:,N-i)+Vs(:,:,N-i)')/2;
    
    Ms_sample_ext(:,N-i)=mvnrnd(Ms(:,N-i),Vs(:,:,N-i));
    
    Ms_sample(:,N-i)=Ms_sample_ext(est_inf,N-i);
    
    
end

%Muestreando condiciones iniciales
J_aux=P_p(:,:,s_sample(1))*R(est_inf,:,s_sample(1))'*(P_pred(est_inf,est_inf,1)^(-1));

%Media de los estados no observables

M0(:)=s_p(:,:,s_sample(1))+J_aux*(Ms_sample(:,1)-M_pred(est_inf,1));

V0(:,:)=P_p(:,:,s_sample(1))-J_aux*(P_pred(est_inf,est_inf,1))*J_aux';

%Forzando simetria (problema de inexactitud de los calculos)

V0(:,:)=(V0(:,:)+V0(:,:)')/2;

s_p(:,:,1)=mvnrnd(M0,V0);

for ot_i=2:m
    
    s_p(:,:,ot_i)=s_p(:,:,ot_i-1);
    
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MUESTREO DE ESTADOS

for i=1:N
    
    %Probabilidades predichas
    
    if i==1
        
        Pr_pred(:,i)=Pr*Pr0(:);
        
    else
        
        Pr_pred(:,i)=Pr*Pr_filt(:,i-1);
        
    end
    
    for h=1:m
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %FILTRO DE HAMILTON
        
        % PREDICCION DE OBSERVABLES
        Mo_pred_aux(:,i,h)=A(est_obs_inf,:,h)*Ms_sample_ext(:,i)+B(est_obs_inf,:,h)*x(i,:)';
        
        % PREDICCION DE NO OBSERVABLES
        
        if i==1
            Mno_pred_aux(:,i,h)=(R(est_inf,:,h)*s_p(:,:,1)+mu(est_inf,:,h));
        else
            Mno_pred_aux(:,i,h)=(R(est_inf,:,h)*Ms_sample_ext(:,i-1)+mu(est_inf,:,h));
        end
        
        %Log verosimilitudes
        
        l_obs(i,h)=-0.5*log(det(T(est_obs_inf,est_obs_inf,h)))-0.5*(y_r(i,est_obs_inf)'-Mo_pred_aux(:,i,h))'*((T(est_obs_inf,est_obs_inf,h)^(-1)))*(y_r(i,est_obs_inf)'-Mo_pred_aux(:,i,h))-size(Mo_pred_aux,1)*log(2*pi)/2;
        l_n_obs(i,h)=-0.5*log(det(K(est_inf,est_inf,h)))-0.5*(Ms_sample(:,i)-Mno_pred_aux(:,i,h))'*((K(est_inf,est_inf,h)^(-1)))*(Ms_sample(:,i)-Mno_pred_aux(:,i,h))-size(Ms_sample,1)*log(2*pi)/2;
        
        l_est(i,:)=l_obs(i,h)+l_n_obs(i,h);
        
    end
    
    
    %Probabilidades filtradas
    Pr_filt(:,i)=exp(l_est(i,:))'.*Pr_pred(:,i)/sum(exp(l_est(i,:))'.*Pr_pred(:,i));
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sampling de estados y smooth condicional

Pr_suav(:,N)=Pr_filt(:,i);
u=rand();
s_sample(N)=sum([u>cumsum(Pr_suav(:,N))])+1;

for i=1:N-1
    
    Pr_suav(:,N-i)=Pr(s_sample(N-i+1),:)'.*Pr_filt(:,N-i)/sum(Pr(s_sample(N-i+1),:)'.*Pr_filt(:,N-i));
    u=rand();
    s_sample(N-i)=sum([u>cumsum(Pr_suav(:,N-i))])+1;
    
end


c_ms=[7,8];
%Ubicacion de los parametros
c_ms_tran=[1,1;2,2];
c_ms_prior_1=[10,10];
c_ms_prior_2=[10,10];


%Ubicacion de los parametros (ecuacion, dim1, dim2, ...)
c_obs_beta=[3,3,3;3,1,1];
c_obs_beta_prior_mean=[0.5,0];
c_obs_beta_prior_var=[0.1,0.1];


c_obs_x_beta=[3,2,2;3,3,3];
c_obs_x_beta_prior_mean=[0.5,0];
c_obs_x_beta_prior_var=[0.1,0.1];


c_nobs_beta=[3,3,3;3,4,4;3,12,0;3,0,12;7,7,7;7,8,8;7,11,0;7,0,11];
c_nobs_beta_prior_mean=[0,0,0,0,0,0,0,0];
c_nobs_beta_prior_var=[1,1,1,1,1,1,1,1];








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MUESTREO DE PARAMETROS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parámetros del MS

for j=1:length(c_ms)
    
    n_c(j,1)=0;
    n_c(j,2)=0;
    
    for i=2:N
        
        if s_sample(i-1)==c_ms_tran(j,1)
            
            if s_sample(i)==c_ms_tran(j,2)
                n_c(j,1) = n_c(j,1) + 1;
            else
                n_c(j,2) = n_c(j,2) + 1;
            end
        end
        
    end
end

c_ms_posterior_1=c_ms_prior_1+n_c(:,1)';
c_ms_posterior_2=c_ms_prior_2+n_c(:,2)';





for j=1:length(c_ms)
    
    Pr(c_ms_tran(j,2),c_ms_tran(j,1))=betarnd(c_ms_posterior_1(j),c_ms_posterior_1(j));
    
end

for h=1:m
    
    Pr(setdiff(1:m,c_ms_tran(c_ms_tran(:,2)==h,1)), h)=1-sum(Pr(c_ms_tran(c_ms_tran(:,2)==h,1), h));
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ecuaciones de observables


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parametros de betas - no observables

if length(c_obs_beta)~=0
    
    
    
    for i_obs_o=1:size(c_obs_beta,1)
        
        %construyendo x e y para obterner las posteriore
        %Se asume que las matrices de errores son diagonales
        
        y_aux=zeros(N,1);
        
        x_aux=zeros(N,1);
        
        var_e=zeros(N,1);
        
        for h=1:m
            
            
            l_o_aux=setdiff(1:size(A,2),c_obs_beta(i_obs_o,1+h));
            l_o_aux_2=intersect(1:size(A,2),c_obs_beta(i_obs_o,1+h));
            y_aux=y_aux+[s_sample(:)==h].*[(A(c_obs_beta(i_obs_o,1),l_o_aux,h)*Ms_sample_ext(l_o_aux,:)+B(c_obs_beta(i_obs_o,1),:,h)*x(:,:)')]';
            if c_obs_beta(i_obs_o,1+h)~=0
                x_aux=x_aux +[s_sample(:)==h].*Ms_sample_ext(l_o_aux_2,:)';
            end
            var_e=var_e + T(c_obs_beta(i_obs_o,1),c_obs_beta(i_obs_o,1),h) * [s_sample(:)==h] .* ones(N,1);
            
        end
        
        y_aux=(y_r(:,c_obs_beta(i_obs_o,1))-y_aux)./var_e;
        x_aux=x_aux./var_e;
        
        %Muestro de parametros
        
        beta_post_m=(c_obs_beta_prior_var(i_obs_o)^-1+x_aux'*x_aux)^(-1)*(c_obs_beta_prior_var(i_obs_o)^-1 *c_obs_beta_prior_mean(i_obs_o) + x_aux' * y_aux);
        beta_post_var=(c_obs_beta_prior_var(i_obs_o)^-1+x_aux'*x_aux)^(-1);
        
        val_sample=normrnd(beta_post_m,beta_post_var);
        
        for h=1:m
            
            if c_obs_beta(i_obs_o,1+h)~=0
                A(c_obs_beta(i_obs_o,1),c_obs_beta(i_obs_o,1+h),h)=val_sample;
            end
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parametros de betas - exógenas


if length(c_obs_x_beta)~=0
    
    %Ecuaciones de observables
    
    %Parametros de betas - no observables
    
    for i_obs_o=1:size(c_obs_x_beta,1)
        
        %construyendo x e y para obterner las posteriore
        %Se asume que las matrices de errores son diagonales
        
        y_aux=zeros(N,1);
        
        x_aux=zeros(N,1);
        
        var_e=zeros(N,1);
        
        for h=1:m
            
            
            l_o_aux=setdiff(1:size(B,2),c_obs_x_beta(i_obs_o,1+h));
            l_o_aux_2=intersect(1:size(B,2),c_obs_x_beta(i_obs_o,1+h));
            y_aux=y_aux+[s_sample(:)==h].*[(A(c_obs_x_beta(i_obs_o,1),:,h)*Ms_sample_ext(:,:)+B(c_obs_x_beta(i_obs_o,1),l_o_aux,h)*x(:,l_o_aux)')]';
            if c_obs_x_beta(i_obs_o,1+h)~=0
                x_aux=x_aux +[s_sample(:)==h].*x(:,l_o_aux_2);
            end
            var_e=var_e + T(c_obs_x_beta(i_obs_o,1),c_obs_x_beta(i_obs_o,1),h) * [s_sample(:)==h] .* ones(N,1);
            
        end
        
        y_aux=(y_r(:,c_obs_x_beta(i_obs_o,1))-y_aux)./var_e;
        x_aux=x_aux./var_e;
        
        %Muestro de parametros
        
        beta_post_m=(c_obs_x_beta_prior_var(i_obs_o)^-1+x_aux'*x_aux)^(-1)*(c_obs_x_beta_prior_var(i_obs_o)^-1 *c_obs_x_beta_prior_mean(i_obs_o) + x_aux' * y_aux);
        beta_post_var=(c_obs_x_beta_prior_var(i_obs_o)^-1+x_aux'*x_aux)^(-1);
        
        val_sample=normrnd(beta_post_m,beta_post_var);
        
        for h=1:m
            
            if c_obs_x_beta(i_obs_o,1+h)~=0
                B(c_obs_x_beta(i_obs_o,1),c_obs_x_beta(i_obs_o,1+h),h)=val_sample;
            end
        end
    end
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parametros betas de ecuaciones de estado


if length(c_nobs_beta)~=0
    
    
    
    for i_obs_o=1:size(c_nobs_beta,1)
        
        %construyendo x e y para obterner las posteriore
        %Se asume que las matrices de errores son diagonales
        
        y_aux=zeros(N-1,1);
        
        x_aux=zeros(N-1,1);
        
        var_e=zeros(N-1,1);
        
        for h=1:m
            
            
            l_o_aux=setdiff(1:size(R,2),c_nobs_beta(i_obs_o,1+h));
            l_o_aux_2=intersect(1:size(R,2),c_nobs_beta(i_obs_o,1+h));
            
            y_aux=y_aux+[s_sample(2:N)==h]'.*([R(c_nobs_beta(i_obs_o,1),l_o_aux,h)*Ms_sample_ext(l_o_aux,1:N-1)]');
            
            if c_nobs_beta(i_obs_o,1+h)~=0
                x_aux=x_aux +[s_sample(2:N)==h]'.*Ms_sample_ext(l_o_aux_2,1:N-1)';
            end
            K
            c_nobs_beta(i_obs_o,1)
            K(c_nobs_beta(i_obs_o,1),c_nobs_beta(i_obs_o,1),h)
            var_e=var_e + K(c_nobs_beta(i_obs_o,1),c_nobs_beta(i_obs_o,1),h) * [s_sample(2:N)==h]' .* ones(N-1,1);
            
        end
        
        
        
        
        
        y_aux=(Ms_sample_ext(c_nobs_beta(i_obs_o,1),2:N)'-y_aux)./var_e;
        
        
        error
        
        
        
        %Muestro de parametros
        
        
        (c_nobs_beta_prior_var(i_obs_o)^-1+x_aux'*x_aux)^(-1)
        
        c_nobs_beta_prior_var(i_obs_o)^-1
        
        x_aux'*x_aux
        
        beta_post_m=(c_nobs_beta_prior_var(i_obs_o)^-1+x_aux'*x_aux)^(-1)*(c_nobs_beta_prior_var(i_obs_o)^-1 *c_nobs_beta_prior_mean(i_obs_o) + x_aux' * y_aux);
        beta_post_var=(c_nobs_beta_prior_var(i_obs_o)^-1+x_aux'*x_aux)^(-1);
        
        val_sample=normrnd(beta_post_m,beta_post_var);
        
        for h=1:m
            
            if c_nobs_beta(i_obs_o,1+h)~=0
                R(c_nobs_beta(i_obs_o,1),c_nobs_beta(i_obs_o,1+h),h)=val_sample;
            end
            
            
        end
    end
    
end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Varianzas de observables

%priors y especificacion
%Primera dim, ecuacion; segunda dim, estado, tercera dim, orden var
c_obs_var=[2,1,1;2,2,2;3,1,1;3,2,2];
c_obs_var_prior=[10,10;10,10;10,10;10,10];


if length(c_obs_var)~=0
    
    c_obs_var_aux=unique(c_obs_var(:,1));
    
    for i_obs_o=1:size(c_obs_var_aux,1)
        
        %construyendo x e y para obterner las posteriore
        %Se asume que las matrices de errores son diagonales
        
        y_aux=zeros(N,1);
        
        var_e=zeros(N,1);
        
        for h=1:m
            
            y_aux=y_aux+[s_sample(:)==h].*[(A(c_obs_var_aux(i_obs_o,1),:,h)*Ms_sample_ext(:,:)+B(c_obs_var_aux(i_obs_o,1),:,h)*x(:,:)')]';
            
            var_e=var_e + T(c_obs_var_aux(i_obs_o,1),c_obs_var_aux(i_obs_o,1)) * [s_sample(:)==h] .* ones(N,1);
            
        end
        
        y_aux=(y_r(:,c_obs_var_aux(i_obs_o,1))-y_aux);
        
        
        %Los parametros que estan en esta ecuacion
        
        c_obs_var_t=c_obs_var(c_obs_var(:,1)==c_obs_var_aux(i_obs_o),:);
        
        %Numero de parametros
        
        c_var_orden=unique(c_obs_var_t(:,3));
        
        
        %Por cada parametro
        for i_ord=1:length(c_var_orden)
            
            %Generando series
            
            c_var=[];
            ratio=[];
            c_var=c_obs_var_t(c_obs_var_t(:,3)==c_var_orden(i_ord),:);
            y_aux_var=zeros(N,1);
            est_param=zeros(N,1);
            
            if i_ord==1
                
                if c_var(1,2)~=0
                    
                    var_h=T(c_var(1,1),c_var(1,1),c_var(1,2));
                    
                else
                    
                    var_h=T(c_var(1,1),c_var(1,1),1);
                    
                end
                
                
            end
            
            if c_var(1,2)~=0
                
                for i_ord2=i_ord:length(c_var_orden)
                    
                    c_var2=[];
                    
                    c_var2=c_obs_var_t(c_obs_var_t(:,3)==c_var_orden(i_ord2),:);
                    
                    ratio(i_ord2-i_ord+1)=T(c_var2(1,1),c_var2(1,1),c_var2(1,2))/T(c_var(1,1),c_var(1,1),c_var(1,2));
                    
                    %Generando series
                    %Estados
                    c_est=c_obs_var_t(c_obs_var_t(:,3)==c_var_orden(i_ord2),2);
                    
                    est=zeros(N,1);
                    
                    for i_est_ord=1:length(c_est)
                        
                        est=est+(s_sample(:)==c_est(i_est_ord));
                        
                    end
                    
                    est_param=est_param+[est>0];
                    
                    if i_ord~=1
                        y_aux_var=y_aux_var+[est>0].*y_aux/((ratio(i_ord2-i_ord+1)*var_h)^0.5);
                        
                    else
                        y_aux_var=y_aux_var+[est>0].*y_aux/(ratio(i_ord2-i_ord+1)^0.5);
                        
                    end
                    
                end
                
            else
                
                y_aux_var=y_aux;
                est_param=ones(N,1);
            end
            
            
            
            for i_param=1:size(c_var,1)
               
                n_param=sum(c_var(i_param,:)==c_obs_var,2)==3;
                
                if i_param==1
                    c_obs_var_posterior1=c_obs_var_prior(n_param,1)+sum(est_param);
                    c_obs_var_posterior2=c_obs_var_prior(n_param,2)+sum(y_aux_var.^2);
                    var_obs_sample=iwishrnd(c_obs_var_posterior1/2,c_obs_var_posterior2/2);
                end
                
                if c_var(1,2)~=0
                
                    T(c_var(1,1),c_var(1,1),c_var(1,2))=var_obs_sample;
                    
                else
                    
                    T(c_var(1,1),c_var(1,1),:)=var_obs_sample;
                    
                end
                
            end
               
        end
       
    end
    
    
    
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Varianzas de no observables

%priors y especificacion
%Primera dim, ecuacion; segunda dim, estado, tercera dim, orden var
c_obs_var=[2,1,1;2,2,2;3,1,1;3,2,2];
c_obs_var_prior=[10,10;10,10;10,10;10,10];

c_nobs_var=[2,1,1;2,2,2;3,1,1;3,2,2];
c_nobs_var_prior=[10,10;10,10;10,10;10,10];



if length(c_nobs_var)~=0
    
    c_nobs_var_aux=unique(c_nobs_var(:,1));
    
    for i_obs_o=1:size(c_nobs_var_aux,1)
        
        %construyendo x e y para obterner las posteriore
        %Se asume que las matrices de errores son diagonales
        
        y_aux=zeros(N-1,1);
        
        var_e=zeros(N-1,1);
        
        
        for h=1:m
            
            y_aux=y_aux+[s_sample(2:N)==h]'.*([R(c_nobs_beta(i_obs_o,1),:,h)*Ms_sample_ext(:,1:N-1)]');
                        
        end
        
        
        
        y_aux=(Ms_sample_ext(c_nobs_beta(i_obs_o,1),2:N)'-y_aux);
        
        
        %Los parametros que estan en esta ecuacion
        
        c_nobs_var_t=c_nobs_var(c_nobs_var(:,1)==c_nobs_var_aux(i_obs_o),:);
        
        %Numero de parametros
        
        c_var_orden=unique(c_nobs_var_t(:,3));
        
        for i_ord=1:length(c_var_orden)
            
            %Generando series
            
            c_var=[];
            ratio=[];
            c_var=c_nobs_var_t(c_nobs_var_t(:,3)==c_var_orden(i_ord),:);
            y_aux_var=zeros(N-1,1);
            est_param=zeros(N-1,1);
            
            if i_ord==1
                
                if c_var(1,2)~=0
                    
                    var_h=K(c_var(1,1),c_var(1,1),c_var(1,2));
                    
                else
                    
                    var_h=K(c_var(1,1),c_var(1,1),1);
                    
                end
                
                
            end
            
            if c_var(1,2)~=0
                
                for i_ord2=i_ord:length(c_var_orden)
                    
                    c_var2=[];
                    
                    c_var2=c_nobs_var_t(c_nobs_var_t(:,3)==c_var_orden(i_ord2),:);
                    
                    ratio(i_ord2-i_ord+1)=K(c_var2(1,1),c_var2(1,1),c_var2(1,2))/K(c_var(1,1),c_var(1,1),c_var(1,2));
                    
                    %Generando series
                    %Estados
                    c_est=c_obs_var_t(c_obs_var_t(:,3)==c_var_orden(i_ord2),2);
                    
                    est=zeros(N-1,1);
                    
                    for i_est_ord=1:length(c_est)
                        
                        est=est+[(s_sample(2:N)==c_est(i_est_ord))]';
                        
                    end
                    
                    est_param=est_param+[est>0];
                    
                    if i_ord~=1
                        y_aux_var=y_aux_var+[est>0].*y_aux/((ratio(i_ord2-i_ord+1)*var_h)^0.5);
                        
                    else
                        y_aux_var=y_aux_var+[est>0].*y_aux/(ratio(i_ord2-i_ord+1)^0.5);
                        
                    end
                    
                end
                
            else
                
                y_aux_var=y_aux;
                est_param=ones(N-1,1);
            end
            
            
            
            
%             
%             
%             for i_param=1:size(c_var,1)
%                
%                 n_param=sum(c_var(i_param,:)==c_obs_var,2)==3;
%                 
%                 if i_param==1
%                     c_obs_var_posterior1=c_obs_var_prior(n_param,1)+sum(est_param);
%                     c_obs_var_posterior2=c_obs_var_prior(n_param,2)+sum(y_aux_var.^2);
%                     var_obs_sample=iwishrnd(c_obs_var_posterior1/2,c_obs_var_posterior2/2);
%                 end
%                 
%                 if c_var(1,2)~=0
%                 
%                     T(c_var(1,1),c_var(1,1),c_var(1,2))=var_obs_sample;
%                     
%                 else
%                     
%                     T(c_var(1,1),c_var(1,1),:)=var_obs_sample;
%                     
%                 end
%                 
%             end
               
        end
       
    end
    
    
    
end






end



