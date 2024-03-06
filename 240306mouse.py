import time

import pyautogui


def auto_click_1():
    """
    在屏幕上自动点击指定位置

    Parameters:
    - x: x坐标
    - y: y坐标
    - clicks: 点击次数，默认为1
    - interval: 点击间隔时间（秒），默认为0.1
    """
    clicks = 1
    interval = 0.2
    target_x = 1071
    target_y = 962
    pyautogui.click(target_x, target_y, clicks=clicks, interval=interval)
    print("1，点了")
    time.sleep(0.5)
    target_x2 = 1068
    target_y2 = 887
    pyautogui.click(target_x2, target_y2, clicks=clicks, interval=interval)
    print("2，点了")
    time.sleep(2)


def auto_click_2():
    """
    在屏幕上自动点击指定位置

    Parameters:
    - x: x坐标
    - y: y坐标
    - clicks: 点击次数，默认为1
    - interval: 点击间隔时间（秒），默认为0.1
    """
    clicks = 1
    interval = 0.2
    target_x = 1752
    target_y = 489
    pyautogui.click(target_x, target_y, clicks=clicks, interval=interval)
    print("列表审核，点了")
    time.sleep(0.5)
    target_x2 = 174
    target_y2 = 821
    pyautogui.click(target_x2, target_y2, clicks=clicks, interval=interval)
    print("选择框，点了")
    time.sleep(0.3)
    target_x3 = 174
    target_y3 = 882
    pyautogui.click(target_x3, target_y3, clicks=clicks, interval=interval)
    print("通过，点了")
    time.sleep(0.3)
    target_x4 = 93
    target_y4 = 941
    pyautogui.click(target_x4, target_y4, clicks=clicks, interval=interval)
    print("确定审核，点了")
    time.sleep(2)


if __name__ == "__main__":

    # 延时5秒，确保你有足够时间将焦点切换到需要点击的窗口
    time.sleep(5)
    type = 'feishu'
    if type == 'feishu':
        # 飞书点击
        for i in range(200):
            auto_click_1()
            print("点击轮次======", i + 1)
    elif type == 'huopin':
        # 网页点击
        for i in range(200):
            auto_click_2()
            print("点击轮次======", i + 1)
    else:
        pass
    print("点完了")
