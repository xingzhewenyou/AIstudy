import time


def find_primes_in_range(start, end, print_interval_multiplier):
    # 初始化一个布尔数组，标记是否为素数
    is_prime = [True] * (end + 1)

    # 0和1不是素数，将其标记为False
    is_prime[0] = is_prime[1] = False

    # 使用埃拉托斯特尼筛法找出素数
    for i in range(2, int(end ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i ** 2, end + 1, i):
                is_prime[j] = False

    prime_count = 0  # 记录素数的计数

    # 打印找到的素数及其位置
    for num in range(max(2, start), end + 1):
        if is_prime[num]:
            prime_count += 1
            decimal_length = len(str(num))
            # print_interval = 10 ** (decimal_length * print_interval_multiplier)
            print_interval = decimal_length * print_interval_multiplier

            if prime_count % print_interval == 0:
                print(f"{prime_count} 素数: {num}")
                time.sleep(0.1)

    print(prime_count)


# 设置寻找的范围
start_range = 9999999
end_range = 1000000000

# 调用函数找出指定范围内的素数并根据十进制长度确定打印间隔
find_primes_in_range(start_range, end_range, print_interval_multiplier=100)
