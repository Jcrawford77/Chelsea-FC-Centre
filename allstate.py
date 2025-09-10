#function question one
def question_1(a, b):
    difference = []
    
    #loop to calculate each element difference from a
    for num in b:
        absolute_difference = abs(a - num)
        difference.append((absolute_difference, num))
    
    #store a list on unique differences in order
    unique_diffs = list(set([d for d, n in difference]))
    unique_diffs.sort(reverse=True)
    #index second element
    second_furthest_difference = unique_diffs[1]
    
    furthest = []
    for diff, num in difference:
        if diff == second_furthest_difference:
            furthest.append(num)
            
    furthest.sort()
    
    return furthest[0]

#function call
result = question_1(22, [19,20,21,24])
print(result)
