function [Hessian,Grad] = hessian_log_ml(c,model,obs,s_p,P_p,x)

    %especifico para MS
    %Varianzas
    var_aux=zeros(9,9);
    var_aux(3,7)=1;
    var_aux(7,3)=1;

    
    n_var=length(c);

    %Derivadas
    %d_LL

    %Hessiano
    %h_LL

    LL=eval(model);
    
    c_back=c;
    var_der=0.001;


    for i1_var=1:n_var



        c=c_back;
        if c(i1_var)==0
            c(i1_var)=c(i1_var)+var_der;
        else
            c(i1_var)=c(i1_var)*(1+var_der);
        end

        LL_aux=eval(model);


        %Calculando derivada

        d_LL(i1_var)=(LL_aux-LL)/(c(i1_var)*var_der);


         for i2_var=1:n_var

%             [i1_var i2_var]

            c=c_back;

            if c(i2_var)==0
                c(i2_var)=c(i2_var)+var_der;
            else
                c(i2_var)=c(i2_var)*(1+var_der);
            end


            LL1_aux=eval(model);

            if c(i1_var)==0
                c(i1_var)=c(i1_var)+var_der;
            else
                c(i1_var)=c(i1_var)*(1+var_der);
            end

            LL2_aux=eval(model);

            d_aux=(LL2_aux-LL1_aux)/(c(i1_var)*var_der);

            h_LL(i2_var,i1_var)=(d_aux-d_LL(i1_var))/(c(i2_var)*var_der);

         end
    end

    Hessian=h_LL;
    Grad=d_LL;
end