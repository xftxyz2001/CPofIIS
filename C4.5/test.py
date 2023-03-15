arr = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 8, 8, 8,
       8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]


def clac_max(arr):
    counter = {}
    for a in arr:
        counter[a] = counter.get(a, 0) + 1
    return counter


def test1():
    c = clac_max(arr)
    print(c)
    print(len(c))

    print(len(arr))


def test2():
    # str -> int
    # 判断是否是数字
    print('123'.isdigit())
    print('123a'.isdigit())
    print('-123'.isdigit())


def isdigit(s):
    try:
        float(s)
        return True
    except:
        return False
    

# 定义一个函数传入一个字符串，如果是整数，返回整数，如果是浮点数，返回浮点数，如果是其他，返回字符串
def str2num(s):
    if isdigit(s):
        if '.' in s:
            return float(s)
        else:
            return int(s)
    else:
        return s



if __name__ == '__main__':
    # test1()
    test2()
