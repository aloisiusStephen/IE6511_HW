function [CostBest] = DDS(x_min,x_max,m,x_initial)
r=0.2;
sBest=x_initial;
sCur=x_initial;
CostBest=bump(sBest);
dimen=length(x_initial);
x_range=x_max-x_min;
k=0;
for i=1:m
    for j=1:dimen
        if (rand(1)<(1-(log(i)/log(m))))
            k=k+1;
            sCur(j)=sCur(j)+randn(1,1)*r*(x_range);
            if(sCur(j)<x_min)
                sCur(j)=x_min+(x_min-sCur(j));
                if(sCur(j)>x_max)
                    sCur(j)=x_min;
                end
            end
            if(sCur(j)>x_max)
                sCur(j)=x_max-(sCur(j)-x_max);
                if(sCur(j)<x_min)
                    sCur(j)=x_max;
                end
            end
        end
    end
    if(k==0)
        index=randi([1,dimen],1);
        sCur(index)=sCur(index)+randn(1,1)*r*(x_range);
            if(sCur(index)<x_min)
                sCur(index)=x_min+(x_min-sCur(index));
                if(sCur(index)>x_max)
                    sCur(index)=x_min;
                end
            end
            if(sCur(index)>x_max)
                sCur(index)=x_max-(sCur(index)-x_max);
                if(sCur(index)<x_min)
                    sCur(index)=x_max;
                end
            end
    end
        k=0;
        if(bump(sCur)>CostBest)
            sBest=sCur;
            CostBest=bump(sBest);
        end
end
end
        
                
                    