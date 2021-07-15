function [b_temp] = temporalFiltering(rmd,band,Smp_Rate)
    order=4;
    [B,A]=butter(order,band/Smp_Rate*2); 
    
    for trl=1:size(rmd,3)
        
        temp=rmd(:,:,trl)';
        ftemp=zscore(filter(B,A,temp));
        b_temp(:,:,trl)=ftemp';
                
    end
        
          
end

