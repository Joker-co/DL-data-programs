'''
sort_maopao() 冒泡排序
时间复杂度：最好O(n2),最差O(n2),平均O(n2)
空间复杂度：最好O(1),最差O(1),平均O(1)
'''

def sort_maopao(nums):
    N = len(nums)
    for i in range(N-1):
        for j in range(N-1-i):
            if nums[j]>nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums

'''
sort_heap() 堆排序
时间复杂度: O(nlogn),O(nlogn),O(nlogn)
空间复杂度: O(1)
'''

def Fixheap(nums, i, length):
    largest = i
    left = i*2+1
    right = (i+1)*2
    
    if left<length and nums[left]>nums[largest]:
        # nums[largest], nums[left] = nums[left], nums[largest]
        largest = left
        
    if right<length and nums[right]>nums[largest]:
        # nums[largest], nums[right] = nums[right], nums[largest]
        largest = right
    
    if largest!=i:
        nums[largest], nums[i] = nums[i], nums[largest]
        Fixheap(nums, largest, length)

def build_heap(nums, length):
    # print(nums)
    for i in range((length//2-1), -1, -1):
        # 调整堆数组
        Fixheap(nums, i, length)

def sort_heap(nums):
    N = len(nums)
    # 构建最大堆
    build_heap(nums, N)
    # 堆排序
    for i in range(N-1,0,-1):
        nums[0], nums[i] = nums[i], nums[0]
        Fixheap(nums, 0, i)
    return nums

nums = [4,1,87,2,7,19,3,11,8]
# print(sort_maopao(nums))
print(sort_heap(nums))

'''
sort_guibin() 归并排序
时间复杂度：最好O(nlogn),最差O(nlogn),平均O(nlogn)
空间复杂度：O(n)
'''
def sort_guibin(data):
    T = len(data)
    temp = data[:]
    
    limit = 1
    while limit<T:
        idx = 0
        while True:
            beg1, end1 = idx, idx + limit -1
            beg2, end2 = idx + limit, idx + 2 * limit -1
            if end1>=T-1:
                break
            if end2>T-1:
                end2 = T-1
            while beg1<=end1 and beg2<=end2:
                if data[beg1]<data[beg2]:
                    temp[idx] = data[beg1]
                    beg1 += 1
                else:
                    temp[idx] = data[beg2]
                    beg2 += 1
                idx += 1
            if beg1<=end1:
                temp[idx] = data[beg1]
                idx += 1
                beg1 += 1
            if beg2<=end2:
                temp[idx] = data[beg2]
                idx += 1
                beg2 += 1
            if idx>=T:
                break
        limit *= 2
        data = temp[:]
    return data

data = [2,3,5,1,88,2,3]
print(sort_guibin(data))
