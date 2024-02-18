from decimal import Decimal, getcontext
import time


def calculate_pi_with_printing(num_digits):
    getcontext().prec = num_digits + 2  # 设置精度，多计算两位以确保准确性
    pi = Decimal(0)

    for k in range(num_digits):
        pi += Decimal((-1) ** k) / Decimal(2 * k + 1)
        approximation = pi * Decimal(4)
        print(f"Approximation after {k + 1} digits: {approximation}")
        time.sleep(1)

    return pi * Decimal(4)


# 设置要计算的 π 的小数位数
num_digits_to_calculate = 1000

# 计算 π 并实时打印每一位的逼近值
start_time = time.time()
result = calculate_pi_with_printing(num_digits_to_calculate)
end_time = time.time()

print(f"\nFinal approximation of pi with {num_digits_to_calculate} decimal places:\n{result}")
print(f"Calculation took {end_time - start_time:.4f} seconds.")
