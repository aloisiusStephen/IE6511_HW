function y = bump(x)

dimen=length(x);

sum1 = 0;
sum2 = 0;
tim1 = 1;
tim2 = 1;
z=zeros(1,dimen);
for i = 1:dimen
    sum1=sum1+(cos(x(i)))^4;
    sum2=sum2+i*(x(i))^2;
    tim1=tim1*(cos(x(i)))^2;
    tim2=tim2*(x(i));
end
for i=1:dimen
    if(0<=x(i)&&x(i)<=10)
        z(i)=0;
    else 
        z(i)=1;
    end
end
if(~any(z)&&tim2>=0.75)
y = abs((sum1-2*tim1)/(sqrt(sum2)));
else
    y=0;
end
end
