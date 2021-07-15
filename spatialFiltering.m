function [s_temp] = spatialFiltering(f_rmd,labels,binClasses)

    labelH=labels==binClasses(1);
    labelF=labels==binClasses(2);
    
    f_rmd_H=f_rmd(:,:,labelH);
    f_rmd_F=f_rmd(:,:,labelF);
    
    f_rmd_H=reshape(f_rmd_H,size(f_rmd_H,1),size(f_rmd_H,2)*size(f_rmd_H,3));
    f_rmd_F=reshape(f_rmd_F,size(f_rmd_F,1),size(f_rmd_F,2)*size(f_rmd_F,3));
    
    [W_CSP] = f_CSP(f_rmd_H,f_rmd_F);
    s_temp=W_CSP;

end

