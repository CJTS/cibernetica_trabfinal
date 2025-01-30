def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], nums[i]]
        seen[num] = nums[i]
    return []

print(two_sum([1,2,3,4,5], 9))