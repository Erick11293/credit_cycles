function [LL,Mf,Vf,Mr,Vr,LLr]=loglkalman(y_r,s_p,P_p,B,x,A,T,R,mu,K)

Mr(1,:)=s_p;
Vr(:,:,1)=P_p;

LL=-length(y_r)*size(y_r,2)*log(2*pi)/2;

%Filtro y cálculo del loglikelihood

for i=1:length(y_r)
  
  
   if i==1
   %Primera recursión
      [Mr(i+1,:),Vr(:,:,i+1),y_p(i,:)]=kalman(B,x(i,:)',A,T,R,K,Vr(:,:,i),Mr(i,:)',y_r(i,:)');
   
     LL=LL-0.5*log(det(A'*(Vr(:,:,i))*A+T))-0.5*(y_r(i,:)'-y_p(i,:)')'*((A'*(Vr(:,:,i))*A+T)^(-1))*(y_r(i,:)'-y_p(i,:)');
     LLr(i)=-0.5*log(det(A'*(Vr(:,:,i))*A+T))-0.5*(y_r(i,:)'-y_p(i,:)')'*((A'*(Vr(:,:,i))*A+T)^(-1))*(y_r(i,:)'-y_p(i,:)')-size(y_r,2)*log(2*pi)/2;
   
   
   else
       
    [Mr(i+1,:),Vr(:,:,i+1),y_p(i,:)]=kalman(B,x(i,:)',A,T,R,K,R*Vr(:,:,i)*R'+K,mu+R*Mr(i,:)',y_r(i,:)');  
    LL=LL-0.5*log(det(A'*(R*Vr(:,:,i)*R'+K)*A+T))-0.5*(y_r(i,:)'-y_p(i,:)')'*((A'*(R*Vr(:,:,i)*R'+K)*A+T)^(-1))*(y_r(i,:)'-y_p(i,:)');
    LLr(i)=-0.5*log(det(A'*(R*Vr(:,:,i)*R'+K)*A+T))-0.5*(y_r(i,:)'-y_p(i,:)')'*((A'*(R*Vr(:,:,i)*R'+K)*A+T)^(-1))*(y_r(i,:)'-y_p(i,:)')-size(y_r,2)*log(2*pi)/2;
  
  end

 
end

%Smoothing

Mf(length(y_r)+1,:)=Mr(length(y_r)+1,:);
Vf(:,:,length(y_r)+1)=Vr(:,:,length(y_r)+1);

for i=1:length(y_r)
  
  J(:,:,length(y_r)-i+1)=Vr(:,:,length(y_r)-i+1)*R'*((R*Vr(:,:,length(y_r)-i+1)*R'+K)^(-1));

  Mf(length(y_r)-i+1,:)=Mr(length(y_r)-i+1,:)'+J(:,:,length(y_r)-i+1)*((Mf(length(y_r)-i+2,:)'-mu-R*(Mr(length(y_r)-i+1,:)')));
  
  Vf(:,:,length(y_r)+1-i)=Vr(:,:,length(y_r)+1-i)+ J(:,:,length(y_r)-i+1)*(Vf(:,:,length(y_r)+2-i)-R*Vr(:,:,length(y_r)-i+1)*R'-K)*J(:,:,length(y_r)-i+1)';

end
 
 
