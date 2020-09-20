def findMaxSubArray(A):
    max_sum = A[0]
    cur_sum = 0
    l = 0
    r = 0
    cur_start = -1
    
    for i in range(len(A)):
        cur_sum+=A[i]
        
        if cur_sum>max_sum:
            
            l = cur_start+1
            r = i
            max_sum = cur_sum
        
        if cur_sum<0:
            
            cur_start = i
            cur_sum = 0
            
    return A[l:r+1]       