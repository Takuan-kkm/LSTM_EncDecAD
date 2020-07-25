import math

a = [3, 4]
b = [1, 3]
c = [-2, -6]

cos = ((a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1])) / (
            math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * math.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2))
print(math.acos(cos)*360/(2*math.pi))
