
%% Example: Discriminant PLS using the NIPALS Algorithm
% Three classes data, each has 50 samples and 4 variables.
clear all;clc;
savepath='D:\ZZZ\External_work\fenglian\ROC_results\Revison3';
datapath='D:\ZZZ\External_work\fenglian';
cd(datapath);
tem=[];num=0;Cov1=[];
[data1 txt raw]=xlsread('BOTH.xlsx',1);
Group=data1(:,2);Cov=data1(:,1:38);

%data2=mapminmax(data1',0,1);
%data1=data2';
Index_Select_MOG_NMO=5:38;

data01=data1(Group==2,Index_Select_MOG_NMO);
data02=data1(Group==3,Index_Select_MOG_NMO);

filename='MS vs HC'

varname=raw(1,Index_Select_MOG_NMO+1);


data11=[data01;data02];

F_th=3;

data11(isnan(data11)==1)=0;
% for i=1:size(data1,2)
%     cov1=[cov_NMOi;cov_MSi];
%     [B Bint R]=regress(data1(:,i),cov1);
%     data1(:,i)=R+mean(data1(:,i));     
% end

y(1:size(data01,1))=1;y(1+size(data01,1):size(data01,1)+size(data02,1))=0;
y=y';
alpha=0.05;
[n m]=size(data11);

% for i=1:m
% [b dev stats]=glmfit(data11(:,i),y,'binomial','link','logit');
% F_value(i)=stats.t(end)^2;
% end
% 
% [F_value IndexF]=sort(F_value,'descend');
% varname=varname(IndexF);
% data11=data11(:,IndexF);

% numz=0;Add_data=[];
% for i=1:m
% [F_Max Index_Max]=max(F_value);
% v1=i;
% v2=n-i-1;
% Fin=finv(1-alpha,v1,v2);
% %Fin=F_th;
% 
% if F_Max<=Fin
%    break;
% else
%    numz=numz+1;
%    Add_data=[Add_data,data11(:,Index_Max)];
%    Sig_var{numz}=varname(Index_Max);
%    varname(Index_Max)=[];
%    data11(:,Index_Max)=[];
%    
%    clear F_value b dev stats F_Max Index_Max Fin
%    for j=1:m-i
%     [b dev stats]=glmfit([Add_data,data11(:,j)],y,'binomial','link','logit');
%     F_value(j)=stats.t(end)^2;
%    end
% %    [F_Max Index_Max]=max(F_value);
% %    v1=i;
% %    v2=n-i-1;
% %    Fin=finv(1-alpha,v1,v2);
% end
% % if numz>=10
% %     break;
% % end
% end
% 
% clear n m;
% [n m]=size(Add_data)


% for i=1:m
% clear F_value b dev stats F_Min Index_Min Fin
% [b dev stats]=glmfit(Add_data,y,'binomial','link','logit')  
% 
% F_value=stats.t(2:end).^2;
% 
% [F_Min Index_Min]=min(F_value);
% 
% v1=1
% v2=n-(m-i+1)-1
% Fout=finv(1-alpha,v1,v2);
% %Fout=F_th;
% if F_Min>Fout
%     break;
% else
%    Sig_var(Index_Min)=[];
%    Add_data(:,Index_Min)=[];
% end
% end
% 
% %[b1 bin1 r1 rint1 stats1]=regress(y,[ones(size(Add_data,1),1),Add_data]);
num=size(data11,1);


for i=1:size(data11,2)
     tem_data=data11(:,i);
     name=varname{i};
  for outer_loop=1:100  
   Per_num=randperm(num,num);
   for j1=1:10
    
   %if j1<11
        Index_test=Per_num(floor(num/10)*(j1-1)+1:floor(num/10)*j1);
        Index_train=Per_num;
        Index_train(floor(num/10)*(j1-1)+1:floor(num/10)*j1)=[];
%     else 
%         Index_test=Per_num(floor(num/10)*(j1-1)+1:num);
%         Index_train=Per_num;
%         Index_train(floor(num/10)*(j1-1)+1:num)=[];
    %end

    
    train_data=tem_data(Index_train);
    test_data=tem_data(Index_test);
    
    train_label=y(Index_train);
    test_label=y(Index_test);
   

[b dev stats]=glmfit(train_data,train_label,'binomial','link','logit')
[yhat dylo dyhi]=glmval(b,test_data,'logit',stats);
[tpr fpr thresholds]=roc(test_label,yhat);
%plotroc
% %计算变量置信区间
%  for per=1:5000
%  idx=randperm(length(y),floor(length(y)*0.9));
%  x1=Add_data(idx,:);y1=y(idx);
%     
%  [b1 dev1 stats1]=glmfit(x1,y1,'binomial','link','logit');
%  CI(per,:)=b1(2:end,1);
%  end
%  
% B_CI5=prctile(CI,5);B_CI95=prctile(CI,95)  
% 
% F_value=stats.t(2:end).^2;
% P_value=stats.p(2:end);
% [F_value Ind_Sig]=sort(F_value,'descend')
% Sig_var=Sig_var(Ind_Sig)
% P_value=P_value(Ind_Sig)
% Sig_var=Sig_var';

for j=1:length(thresholds)
threhold(j,1)=thresholds{j}(2);
end

% threhold=X;
threhold=sort(threhold);



for j=1:length(threhold)
yhat1=yhat;
yhat1(yhat>=threhold(j))=1;
yhat1(yhat<threhold(j))=0;

% yhat1=y;
% yhat1(X>=threhold(j))=1;
% yhat1(X<threhold(j))=0;
num1=length(find(test_label==1));
num0=length(find(test_label==0));

TP=sum(yhat1.*test_label);
TN=sum((1-yhat1).*(1-test_label));
FP=num0-TN;
FN=num1-TP;
TPR=TP/(TP+FN);
FPR=FP/(FP+TN);

TPRZ(j)=TPR;
FPRZ(j)=FPR;
Yoden(j)=TPR-FPR;
Acc(j)=(TP+TN)/length(yhat1);
Refx(j)=1/length(threhold)*j;Refy(j)=1/length(threhold)*j;
end
%[cut_Yoden zz]=max(Acc);
[cut_Yoden zz]=max(Yoden);
cut_value=yhat(zz);

auc=-trapz(FPRZ,TPRZ);
% h1=figure;
% plot(FPRZ,TPRZ,'b--');
% hold on;
% plot(FPRZ(zz),TPRZ(zz),'rO');
% hold on;
% plot(Refx,Refy,'k-');
% xlabel('FDR');ylabel('TPR');title(['ROC of ',filename,'(AUC=',num2str(auc),')']);

% if auc>0.6
% cd(savepath);
% imgname=['ROC_of_',filename,'.fig'];
% savefig(h1,imgname);
% end
% %pause(2);
close all;
% list{1}=filename;
% list{2}=cut_Yoden;
% list{3}=cut_value;
% list{4}=TPRZ(zz);
% list{5}=1-FPRZ(zz);
% list{6}=Acc(zz);
list(i,outer_loop,j1)=auc;

  end
  end
end
% list{9}=stats.beta(2:end);
% list{10}=B_CI5;
% list{11}=B_CI95;
% list{12}=stats.p(2:end);

list1=reshape(list,size(data11,2),outer_loop*10);
list1(isnan(list1))=0;

list2(:,1)=mean(list1,2);
list2(:,2)=std(list1,0,2);
list2(:,3)=median(list1,2);
list2(:,4)=prctile(list1,25,2);
list2(:,5)=prctile(list1,75,2);
list2(:,6)=list2(:,5)-list2(:,4);


cd(savepath);
xlswrite(['ROC_results_',filename,'.xlsx'],list2);