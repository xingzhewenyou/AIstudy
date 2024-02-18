from ecpy.curves import Curve, Point
from ecpy.keys import ECPublicKey
from ecpy.ecdsa import ECDSA


def ecm_factorization(N, curve_params):
    # 创建椭圆曲线
    curve = Curve.get_curve('P-256')

    # 选择起始点
    x = 2  # 可以选择不同的初始值
    y_squared = (x ** 3 + curve_params['a'] * x + curve_params['b']) % curve.field
    y = pow(y_squared, (curve.field + 1) // 4, curve.field)  # 平方根
    P = Point(x, y, curve)

    # 尝试找到非平凡因子
    for i in range(1, 1000):  # 尝试次数，可以根据需求调整
        r = i  # 随机选择一个整数
        Q = r * P  # 点的倍乘运算

        gcd_value = ECDSA.gcd(Q.x, N)
        if 1 < gcd_value < N:
            return gcd_value, N // gcd_value  # 找到非平凡因子

    return None  # 未找到非平凡因子


# 待分解的合数 N
N = 5959  # 这里只是一个示例，你可以替换成你需要分解的合数

# 椭圆曲线参数（注意：这里的参数 a 和 b 需要根据所选椭圆曲线的具体参数进行调整）
curve_params = {'a': 0, 'b': 7}

# 运行 ECM 算法
result = ecm_factorization(N, curve_params)

if result:
    p, q = result
    print(f"{N} 的非平凡因子为：{p} 和 {q}")
else:
    print(f"在尝试范围内未找到 {N} 的非平凡因子")
