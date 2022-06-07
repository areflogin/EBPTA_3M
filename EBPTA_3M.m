%   Author:  Aref YELGHI                                     
%   year:    2013                                                  
%   E-mail:  ar.yelqi (at) gmail (dot) com  

clc; close all; clear all;

% create training set.
cat=3;
num=3;

ginput(1);

for i=1:cat
   for j=1:num
       [x,y,button] = ginput(1);
         pat_1x(j)=x;
         pat_1y(j)=y;
      if i==1
         pat_1=[pat_1x; pat_1y];
         plot(x,y,'*r','MarkerSize',12); 
      end
      if i==2
          pat_2=[pat_1x; pat_1y];
         plot(x,y,'+b','MarkerSize',12); 
      end
      if i==3
          pat_3=[pat_1x; pat_1y];
         plot(x,y,'xg','MarkerSize',12);      
      end
%       if i==4
%           pat_4=[pat_1x; pat_1y];
%          plot(x,y,'ok','MarkerSize',12);%,'MarkerSize',25); 
%       end
%       if i==5
%           pat_5=[pat_1x; pat_1y];
%          plot(x,y,'oy','MarkerSize',12);%,'MarkerSize',25); 
%       end
         axis([0 1 0 1])
         hold on 
   end
end

title('Two-dimensional Multi-class pattern space (Non-Linear Classification)');
xlabel('x');
ylabel('y');


% augmented input vectors
pat_1(3,:)=-1; % adding third column 1 to main data
y1=pat_1;
d1=-1*ones(cat,num);
d1(1,:)=d1(1,:)+2*ones(1,num);

pat_2(3,:)=-1; 
y2=pat_2;
d2=-1*ones(cat,num);
d2(2,:)=d1(2,:)+2*ones(1,num);

pat_3(3,:)=-1; 
y3=pat_3;
d3=-1*ones(cat,num);
d3(3,:)=d1(3,:)+2*ones(1,num);

% pat_4(3,:)=-1; 
% y4=pat_4;
% d4=-1*ones(cat,num);
% d4(4,:)=d1(4,:)+2*ones(1,num);
% 
% pat_5(3,:)=-1; 
% y5=pat_5;
% d5=-1*ones(cat,num);
% d5(5,:)=d1(5,:)+2*ones(1,num);

z=[y1 y2 y3];% y4 y5];
d=[d1 d2 d3];% d4 d5];



% RDPTA Implementation
% RDPTA Implementation
%  
 I=3; 
 J=6; 
 K=3; 
 y=zeros(J,1);
 y(J,1)= -1;
 
  n=0.1;
  L=1;
  W= (-1+2*rand(J,K));
  V=(-1+2*rand(I,J-1));
  Emax=0.1; 
  E=0;
  p=1;
  q=1;

  W
  V
  
while q<1000
   b1=0; b2=0;% b3=0; b4=0;
   
   while p <= (cat*num)
       % compute y
       for j=1:J-1
           nety=V(:,j)'*z(:,p);
           y(j)=(2/(1+exp(-L*nety)))-1;
           
           m0(j)=y(j);
       end
       if p<=3
           m1(:,p)=[m0];
       end 
       
        if (p>3) && (p<=6) 
           b1=b1+1;
           m2(:,b1)=[m0];
        end   
        
       if (p>6) % && (p<=6) 
           b2=b2+1;
           m3(:,b2)=[m0];
       end        
       
%        if  (p>6) && (p<=8) 
%            b3=b3+1;
%            m4(:,b3)=[m0];
%        end        
%        
%        if (p>8) 
%            b4=b4+1;
%            m5(:,b4)=[m0];
%        end        
       
       % compute o
       for k=1:K
           neto=W(:,k)'*y;
           o(k)=(2/(1+exp(-L*neto)))-1;   
           E=0.5*((d(k,p)-o(k)).^2)+E;
           
           So(k)= 0.5*(d(k,p)-o(k))*(1-o(k).^2); % output layer error signal
       end
           % hidden layer error signal 
       for j=1:J-1
           SS=0;
           for k=1:K
               SS=SS+ So(k)*W(j,k);
           end
           Sy(j)= 0.5*(1-y(j).^2)* SS; % hidden layer error signal
       end       
          % W and V Weights update      
           for k=1:K
               for j=1:J
                  W(j,k)=W(j,k)+ n * So(k) * y(j);  % output layer weight adjusted
               end
           end
               for j=1:J-1
                  for i=1:I
                       V(i,j)=V(i,j)+ n * Sy(j) * z(i,p);  % hidden layer weight adjusted
                  end
               end
      
       p=p+1;
   end
   
   Eu(q)=E;
%     % plot line
% if k==1
%      % ax + by + c = 0 ==> y=(-c-ax)/b;
%      x1 = [0,1];   %x = [xmin,xmax]; 
%      if W(1,2)==0
%          W(1,2)=W(1,2)+0.0001;
%      end
%      y1 = (-W(1,3)/W(1,2))-((W(1,1)*x1)/W(1,2));
%      plot(x1,y1,'Color','r')
%      hold on

%      
%      if W(2,2)==0
%          W(2,2)=W(2,2)+0.0001;
%      end
%      y2 = (-W(2,3)/W(2,2))-((W(2,1)*x1)/W(2,2));
%      plot(x1,y2,'Color','b')
%      hold on
%      
%      if W(3,2)==0
%          W(3,2)=W(3,2)+0.0001;
%      end
%      y3 = (-W(3,3)/W(3,2))-((W(3,1)*x1)/W(3,2));
%      plot(x1,y3,'Color','g')
%      hold on
%      
%      if W(4,2)==0
%          W(4,2)=W(4,2)+0.0001;
%      end
%      y4 = (-W(4,3)/W(4,2))-((W(4,1)*x1)/W(4,2));
%      plot(x1,y4,'Color','k')
%      hold on
%     
%     if W(4,2)==0
%          W(4,2)=W(4,2)+0.0001;
%      end
%      y5 = (-W(5,3)/W(5,2))-((W(5,1)*x1)/W(5,2));
%      plot(x1,y5,'Color','y')
%      hold on
%   drawnow; pause(0.1)
% end 
   
   if E<Emax
     break
   else
       E=0;
       p=1;
   end
 q=q+1;
end
%
 q
 W
 V
 
 V=V'; 
     x1 = [-1,1];   %x = [xmin,xmax]; 
     if V(1,2)==0
         V(1,2)=V(1,2)+0.0001;
     end
     y1 = (V(1,3)/V(1,2))-((V(1,1)*x1)/V(1,2));
     plot(x1,y1,'Color','r')
     hold on
     
     if V(2,2)==0
         V(2,2)=V(2,2)+0.0001;
     end
     y2 = (V(2,3)/V(2,2))-((V(2,1)*x1)/V(2,2));
     plot(x1,y2,'Color','b')
     hold on
     
     if V(3,2)==0
         V(3,2)=V(3,2)+0.0001;
     end
     y3 = (V(3,3)/V(3,2))-((V(3,1)*x1)/V(3,2));
     plot(x1,y3,'Color','g')
     hold on
     
     if V(4,2)==0
         V(4,2)=V(4,2)+0.0001;
     end
     y4 = (V(4,3)/V(4,2))-((V(4,1)*x1)/V(4,2));
     plot(x1,y4,'Color','k')
     hold on
    
    if V(5,2)==0
         V(5,2)=V(5,2)+0.0001;
     end
     y5 = (V(5,3)/V(5,2))-((V(5,1)*x1)/V(5,2));
     plot(x1,y5,'Color','y')
     hold on
     
figure (2)

plot(m1(1,:),m1(2,:),'*r','MarkerSize',12); 
axis([-1 1 -1 1])
hold on

plot(m2(1,:),m2(2,:),'+b','MarkerSize',12); 
hold on

plot(m3(1,:),m3(2,:),'xg','MarkerSize',12); 
hold on

% plot(m4(1,:),m4(2,:),'ok','MarkerSize',12); 
% hold on
% 
% plot(m5(1,:),m5(2,:),'oy','MarkerSize',12); 
% hold on

% %W=W'; 
% x1 = [-1,1];   %x = [xmin,xmax]; 
% if W(1,2)==0
%       W(1,2)=W(1,2)+0.0001;
% end
% y1 = (W(1,3)/W(1,2))-((W(1,1)*x1)/W(1,2));
% plot(x1,y1,'Color','r')
% hold on
%      
% if W(2,2)==0
%     W(2,2)=W(2,2)+0.0001;
% end
% y2 = (W(2,3)/W(2,2))-((W(2,1)*x1)/W(2,2));
% plot(x1,y2,'Color','b')
% hold on
%      
% if W(3,2)==0
%      W(3,2)=W(3,2)+0.0001;
% end
% y3 = (W(3,3)/W(3,2))-((W(3,1)*x1)/W(3,2));
% plot(x1,y3,'Color','g')
% hold on
%      
% if W(4,2)==0
%      W(4,2)=W(4,2)+0.0001;
% end
% y4 = (W(4,3)/W(4,2))-((W(4,1)*x1)/W(4,2));
% plot(x1,y4,'Color','k')
% hold on
%     
% if W(5,2)==0
%     W(5,2)=W(5,2)+0.0001;
% end
% y5 = (W(5,3)/W(5,2))-((W(5,1)*x1)/W(5,2));
% plot(x1,y5,'Color','y')
% hold on   
% 
% if W(6,2)==0
%     W(6,2)=W(6,2)+0.0001;
% end
% y6 = (W(6,3)/W(6,2))-((W(6,1)*x1)/W(6,2));
% plot(x1,y6,'Color','y')
% hold on   
     
title('Space Transformation');
xlabel('y1');
ylabel('y2');    
%      
 t=1:length(Eu);
 figure (4)
 plot(t,Eu,'Color','r')
 title('Error Convergence');
 xlabel('iteration number (k)');
 ylabel('Error');
 

