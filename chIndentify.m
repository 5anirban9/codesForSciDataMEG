function [ chIndentity ] = chIndentify( allChann )
%%%%select a specific brain area for analysis%%%%
%%Identifiers for specific brain areas are as follows:
%%Motor Cortex=MCids
%%FrontalLobe=Fids
%%LeftTemporalLobe=LTids
%%RightTemporalLobe=RTids
%%occipitalLobe=Oxids

 MCids={'064','062','103','041','042','063','104','111','112','044','043','071','072','114','113','181','182','074','073','221','222','183','224'}';
 LTids={'031','011','012','034','032','033','013','021','022','014','151','024','023','154','152','161','162','153','172','164','163'}';
 RTids={'121','124','123','122','141','131','132','144','142','134','133','261','143','262','241','242','264','263','244','243','252'}';
 Fids={'052','081','091','051','053','082','094','092','054','061','101','102','093'}';
 Occids={'184','201','202','223','191','204','203','231','194','192','211','234','232','173','193','212','233','251','174','214','213','254'}';

% allChann=MEGdata.c;

% index = find(strcmp(allChann, 'MEG0123'));
chIndentity=zeros(size(allChann,1),2);

for ch=1:length(allChann)
    strfull = allChann{ch};
    str=strfull(4:6);
    if(~isempty(find(strcmp(MCids, str), 1)))
        chIndentity(ch,:)=[1 str2double(strfull(7))];
    elseif(~isempty(find(strcmp(LTids, str), 1)))
        chIndentity(ch,:)=[2 str2double(strfull(7))];
    elseif(~isempty(find(strcmp(RTids, str), 1)))
        chIndentity(ch,:)=[3 str2double(strfull(7))];
    elseif(~isempty(find(strcmp(Fids, str), 1)))
        chIndentity(ch,:)=[4 str2double(strfull(7))]; 
    elseif(~isempty(find(strcmp(Occids, str), 1)))
        chIndentity(ch,:)=[5 str2double(strfull(7))];       
    end
end
      


%Description of the numbers inside 'chIndentity'
%Most significant bit
%MotorCortex=1, LeftTemporalLobe=2, RightTemporalLobe=3,
%FrontalLobe=4,OccipitalLobe=5
%Least significant bit
%Magnetometer=1, gradiometer=2,3; latitude/longitude depends on the %location







