def GD(sInitial, maxIter):
    
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
        
        # Find a neighbor
        start = np.maximum(sCurrent-10, 0)
        end = np.minimum(sCurrent+10, 500)

        best_neighbor_cost = cost(start)
        best_neighbor = start
        
        for x in range(start, end+1):
            curr_neighbor_cost = cost(x)
            if curr_neighbor_cost < best_neighbor_cost:
                best_neighbor = x
                best_neighbor_cost = curr_neighbor_cost

        # End of finding a neighbor    
        
        sCurrent = best_neighbor
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