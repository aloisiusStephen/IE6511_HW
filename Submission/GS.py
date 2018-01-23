def GS(sInitial, maxIter):
    
    iter_num = []
    current_s = []
    best_s = []
    current_cost = []
    best_cost = []
    
    i = 1
    sCurrent = sInitial
    sBest = sInitial
    costCurrent = cost(sInitial)
    costBest = costCurrent
    
    while i <= maxIter:
        
        iter_num.append(i)
        best_s.append(sBest)
        best_cost.append(costBest)
        
        sCurrent = neighbor(sBest)
        costCurrent = cost(sCurrent)
        if costCurrent < costBest:
            costBest = costCurrent
            sBest = sCurrent

        current_s.append(sCurrent)
        current_cost.append(costCurrent)

        i+=1
        
    Solution = pd.DataFrame({"i": iter_num, "sCurrent": current_s, "sBest": best_s, 
                             "costCurrent": current_cost, "costBest": best_cost})
    return Solution