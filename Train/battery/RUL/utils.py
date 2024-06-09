def closest_values(data, target_value: float, cycle: int, phase: bool):
    """
    Approximating the value of ´x´ for a given value of ´y´
    of a function f(x)=y, and get the time of 'x'

    Parameters
    ----------
    data : dataframe.
    target_value : the value of y (desired voltage value).
    cycle : cycle of the battery.
    phase : bool
        1 for discharging phase, 0 for charging phase.

    Returns
    -------
    feature : the time (s) of x in the given cycle

    """

    if phase == 0:
        df_filt = data[data['Current'] > 0]  # 选择充电阶段

    elif phase == 1:
        df_filt = data[data['Current'] < 0]  # 选择放点阶段

    df_filt = df_filt[df_filt['Cycle_Index'] == cycle]  # Selects battery cycle

    # 寻找最接近target_value的值
    a_list = list(df_filt['Voltage'])
    absolute_difference_function = lambda list_value: abs(list_value - target_value)

    closest_value = min(a_list, key=absolute_difference_function)
    ind_1 = a_list.index(closest_value)  # 获取最近的位置
    time_closest_value = df_filt.reset_index()['Test_Time'][ind_1]  # 最接近target_value的时间

    cycle_start = df_filt['Test_Time'].min()  # 充电开始时间

    if closest_value == target_value:
        # 充/放电的时间长度作为特征
        feature = time_closest_value - cycle_start

    else:  # Finding the value between two points
        b_list = a_list.copy()
        b_list.remove(closest_value)
        second_closest_value = min(b_list, key=absolute_difference_function)  # 第二接近最接近target_value的时间的值
        ind_2 = a_list.index(second_closest_value)
        time_second_closest_value = df_filt.reset_index()['Test_Time'][ind_2]  # 同理，获取时间

        y1 = time_closest_value
        y2 = time_second_closest_value
        x1 = closest_value
        x2 = second_closest_value
        x = target_value

        if closest_value < second_closest_value:
            y = y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)  # 计算一个线性近似值

        else:
            y = y2 + ((x - x2) / (x1 - x2)) * (y1 - y2)  # 计算一个线性近似值

        feature = y - cycle_start

    return feature
