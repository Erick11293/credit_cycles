function [LL,M_pred,P_pred,Pr_pred,Mr,Vr,Pr_filt,Ms,Vs,Pr_s,Mss,l_p_obs]= loglkalmanms(y_r,s_p,P_p,B,x,A,T,R,mu,K,Pr,Pr0,suav,vinic)
%Representación estado espacio del modelo
%y=As+Bx+e
%s=mu+Rs(-1)+v

%E(e'e)=T
%E(v'v)=K
%las matirces A, B, R, T, K, s_p, P_p edben ser de tres dimensiones. LA
%tercera dimension corresponde a la variabilidad de parametros del MS. La
%matriz Pr debe de ser de nxn donde n es el número de estados del MS
%Pr0 es la matriz de probabilidades inicial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=length(Pr);
N=length(y_r);
n_no=size(R,2);
n_o=size(y_r,2);


%Reservando espacio en memoria
LL=0;
Pr_pred=zeros(m,m,N);
M_pred=zeros(n_no,N,m,m);
P_pred=zeros(n_no,n_no,N,m,m);

Mr_aux=zeros(n_no,N,m,m);
Vr_aux=zeros(n_no,n_no,N,m,m);
y_p=zeros(n_o,N,m,m);
var_p=zeros(n_o,n_o,N,m,m);

Pr_filt=zeros(m,N);

Mr=zeros(n_no,N,m);
Vr=zeros(n_no,n_no,N,m);


Pr_aux=zeros(m,m,N);

Pr_s_aux=zeros(m,m,N);
Ms_aux=zeros(n_no,m,m,N);
Ps_aux=zeros(n_no,n_no,m,m,N);

l_p_obs=zeros(N,1);

Ms=zeros(n_no,N,m);
Vs=zeros(n_no,n_no,N,m);
Mss=zeros(n_no,N);
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

%Probabilidad en cada estado


%Numero de estados 
    

LL=0;




%Matrices

%Pr_pred(1:m,1:m,1:N)=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FILTRO Y CÁLCULO DEL LOGLIKELIHOOD

for i=1:length(y_r)
    
    %PASO 1 - PREDECIR PROBABILIDADES
    %Probabilidades predichas del periodo
    
    for j=1:m
        
        for h=1:m

            if i==1

                Pr_pred(h,j,i)=Pr(h,j)*Pr0(j);

            else

                Pr_pred(h,j,i)=Pr(h,j)*Pr_filt(j,i-1);

            end

    
        end
        
    end
    %PASO 2 - FILTRAR ESTADOS Y CALCULAR VEROSIMILITUD POR ESTADOS
    
    %Filtro de kalman para cada estado en t | cada estado en j-1
    %Estado anterior
    for j=1:m
           
     
        %Estado actual
        for h=1:m
            
            %Medias predichas y varianzas de la proyeccion
            
            if i==1
                M_pred(:,i,h,j)=(R(:,:,h)*s_p(:,:,j)+mu(:,:,h));
                P_pred(:,:,i,h,j)=R(:,:,h)*P_p(:,:,j)*R(:,:,h)'+K(:,:,h); 
               
            else
                M_pred(:,i,h,j)=(R(:,:,h)*Mr(:,i-1,j)+mu(:,:,h));
                
                P_pred(:,:,i,h,j)=R(:,:,h)*Vr(:,:,i-1,j)*R(:,:,h)'+K(:,:,h); 
                
            end
                
           %Filtro de kalman  
            
            [Mr_aux(:,i,h,j),Vr_aux(:,:,i,h,j),y_p(:,i,h,j),var_p(:,:,i,h,j)]=kalman(B(:,:,h),x(i,:)',A(:,:,h),T(:,:,h),P_pred(:,:,i,h,j),M_pred(:,i,h,j),y_r(i,:)');
            
            %Calculando verosimilitud
            
            LLr_aux(i,h,j)=-0.5*log(det(var_p(:,:,i,h,j)))-0.5*(y_r(i,:)'-y_p(:,i,h,j))'*((var_p(:,:,i,h,j)^(-1)))*(y_r(i,:)'-y_p(:,i,h,j));
            %-size(y_r,2)*log(2*pi)/2;
            
            %Verosimilitud por la probabilidad predicha
            
            LLr(i,h,j)=LLr_aux(i,h,j)+log(Pr_pred(h,j,i));
        end
        
        
    end
    
    
    %PASO 3 - VEROSIMILITUD
    
    %probabilidad de observar i
    
    
    l_p_obs(i)=log(sum(sum(exp(LLr(i,:,:)))));
    
    %PASO 4 - ACTUALIZAR PROBABILIDADES
    
    for j=1:m
        
        for h=1:m
            
            Pr_aux(h,j,i)=exp(LLr(i,h,j))/exp(l_p_obs(i));
                     
        end
        
    end
    
    for h=1:m
        
        Pr_filt(h,i)=squeeze(sum(Pr_aux(h,:,i),2));
    
    end
    
    
    %PASO 5 - COLAPSAR VARIABLES
    
    %La media
    
    for h=1:m
        
        Mr(:,i,h)=reshape(Mr_aux(:,i,h,:),size(R,2),m)*Pr_aux(h,:,i)'/Pr_filt(h,i);
        
        Vr(:,:,i,h)=0;
       
        for j=1:m
            
            Vr(:,:,i,h)=Vr(:,:,i,h)+Pr_aux(h,j,i)*(Vr_aux(:,:,i,h,j)+(Mr(:,i,h)-Mr_aux(:,i,h,j))*(Mr(:,i,h)-Mr_aux(:,i,h,j))')/Pr_filt(h,i);

        end     
              
        
    end      
    
end

LL=sum(l_p_obs);


if suav==true

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SUAVIZADO


    %Ultimo periodo

    T=size(y_r,1);

    Ms(:,T,:)=Mr(:,T,:);
    Vs(:,:,T,:)=Vr(:,:,T,:);
    Pr_s(:,T)=Pr_filt(:,T);

    Mss(:,T)=reshape(Ms(:,T,:),size(R,2),m)*Pr_s(:,T);
    
    
    for i=1:T-1
       
        for j=1:m

            for h=1:m

                %Probabilidades suavizadas
                Pr_s_aux(h,j,T-i)=Pr_s(h,T-i+1)*Pr_pred(h,j,T-i+1)/sum(Pr_pred(h,:,T-i+1));
       
                J_aux=Vr(:,:,T-i,j)*R(:,:,h)'*(P_pred(:,:,T-i+1,h,j)^(-1));
                %Medias suavizadas, cov suavizada

                Ms_aux(:,h,j,T-i)=Mr(:,T-i,j)+J_aux*(Ms(:,T-i+1,h)-M_pred(:,T-i+1,h,j));
                
             
                
                Ps_aux(:,:,h,j,T-i)=Vr(:,:,T-i,j)+J_aux*(Vs(:,:,T-i+1,h)-P_pred(:,:,T-i+1,h,j))*J_aux';
                

            end
    
            %Colapsando probabilidades
            Pr_s(j,T-i)=sum(Pr_s_aux(:,j,T-i));
            

            Ms(:,T-i,j)=Ms_aux(:,:,j,T-i)*Pr_s_aux(:,j,T-i)/Pr_s(j,T-i);
            
            Vs(:,:,T-i,j)=0;

                for h=1:m

                    Vs(:,:,T-i,j)=Vs(:,:,T-i,j)+Pr_s_aux(h,j,T-i)*(Ps_aux(:,:,h,j,T-i)+(Ms(:,T-i,j)-Ms_aux(:,h,j,T-i))*(Ms(:,T-i,j)-Ms_aux(:,h,j,T-i))')/Pr_s(j,T-i);
                    
                end

        end
                
        Mss(:,T-i)=reshape(Ms(:,T-i,:),size(R,2),m)*Pr_s(:,T-i);
        
    end
    
    
end
    