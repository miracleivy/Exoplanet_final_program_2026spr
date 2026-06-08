from skyfield.api import load
import numpy as np
import matplotlib.pyplot as plt

def calendar_to_jd(year, month, day, calendar_type="julian"):
    """将指定的历法日期转换为连续的儒略日 (JD)"""
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    if calendar_type.lower() == "gregorian":
        B = 2 - A + int(A / 4)
    else:
        B = 0
    jd = (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + B
        - 1524.5
    )
    return jd


def jd_to_calendar(jd, to_calendar="gregorian"):
    """将中转的儒略日 (JD) 准确转换回目标历法日期"""
    jd += 0.5
    Z = int(jd)
    F = jd - Z
    if to_calendar.lower() == "gregorian":
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)
    else:
        A = Z
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day = B - D - int(30.6001 * E) + F
    if E < 14:
        month = E - 1
    else:
        month = E - 13
    if month > 2:
        year = C - 4716
    else:
        year = C - 4715
    return int(year), int(month), int(day)


def julian_to_gregorian_preset(year, month, day):
    """主转换核心函数：儒略历 转 格里高利历"""
    jd_mid = calendar_to_jd(year, month, day, calendar_type="julian")
    g_year, g_month, g_day = jd_to_calendar(jd_mid, to_calendar="gregorian")
    return g_year, g_month, g_day

def check_totality(test_dt,ts, eph, qufu, sun, moon, year, m, d, ut1_hours):
    # A.  TT = UT1 + Delta_T
    # 我们先得到一系列 TT 的儒略日
    t_temp = ts.ut1(year, m, d, ut1_hours)
    # 计算对应的 TT 儒略日：当前的 UT1 + 我们要测试的 Delta T
    jd_tt = t_temp.ut1 + test_dt / 86400.0
    
    # B. 使用官方推荐的 tt_jd 构造时间对象
    # 此时生成的 t 对象，其内置的 delta_t 是 Skyfield 用默认模型算出来的
    t = ts.tt_jd(jd_tt) 
    
    # C. 【核心黑客操作】强制覆盖该对象的 delta_t 属性
    # Skyfield 计算地面位置时会读取 t.delta_t
    t.delta_t = np.array([test_dt] * len(jd_tt))
    
    # 3. 几何判定
    obs_s = qufu.at(t).observe(sun).apparent()
    obs_m = qufu.at(t).observe(moon).apparent()
    
    sep = obs_s.separation_from(obs_m).radians
    # 使用 IAU 常数和更精确的半径计算
    r_sun = np.arcsin(695700.0 / obs_s.distance().km)
    r_moon = np.arcsin(1737.4 / obs_m.distance().km)
    
    is_bigger = r_moon > r_sun
    is_covered = sep < (r_moon + r_sun) 
    #- (2 * r_moon) * (1 / 2) # 缩小范围
    
    return np.any(is_bigger & is_covered)

def check_totality_updated(test_dt, ts, eph, qufu, sun, moon, year, m, d, ut1_hours_coarse):
    """
    ΔT扫描运用【粗筛+精筛】方法
    注意：为了实现粗筛，最后一个参数传入的是粗颗粒度的时间网格（例如每半小时采样一次）。
    """
    # === 【第一阶段：粗搜索】快速定位日食可能发生的“小时” ===
    
    # 1. 用每半小时一次的粗颗粒度网格，生成当天的 UT1 时间对象
    t_coarse_temp = ts.ut1(year, m, d, ut1_hours_coarse)
    
    # 2. 结合当前测试的 test_dt，换算得到粗筛下的 TT 儒略日
    jd_tt_coarse = t_coarse_temp.ut1 + test_dt / 86400.0
    t_coarse = ts.tt_jd(jd_tt_coarse)
    
    # 3. 关键步骤：强制覆盖 delta_t
    t_coarse.delta_t = np.array([test_dt] * len(jd_tt_coarse))
    
    # 4. 计算粗筛下的日月视位置与角距离，弧度
    obs_s_c = qufu.at(t_coarse).observe(sun).apparent()
    obs_m_c = qufu.at(t_coarse).observe(moon).apparent()
    seps_coarse = obs_s_c.separation_from(obs_m_c).radians
    
    # 5. 锁定日月最贴近的粗筛网格索引，提取中心小时
    best_coarse_idx = np.argmin(seps_coarse)
    center_hour = ut1_hours_coarse[best_coarse_idx]
    
    
    # === 【第二阶段：精搜索】在锁定的时间前后，用 1 秒步长进行几何判定 ===
    
    # 6. 以中心时间为基准，前后扩展 0.6 小时，建立 1 秒步长的精细网格
    ut1_hours_fine = np.arange(center_hour - 0.6, center_hour + 0.6, 1.0 / 3600.0)
    
    # 7. 用精细网格重新生成时间对象并注入 test_dt
    t_temp = ts.ut1(year, m, d, ut1_hours_fine)
    jd_tt = t_temp.ut1 + test_dt / 86400.0
    t = ts.tt_jd(jd_tt)
    t.delta_t = np.array([test_dt] * len(jd_tt))
    
    # 8. 计算 1 秒高精度步长下的太阳、月亮精确视位置
    obs_s = qufu.at(t).observe(sun).apparent()
    obs_m = qufu.at(t).observe(moon).apparent()
    
    # 9. 计算高精度角距离和两者的视半径（底层逻辑与原几何判定完全相同）
    sep = obs_s.separation_from(obs_m).radians
    r_sun = np.arcsin(695700.0 / obs_s.distance().km)
    r_moon = np.arcsin(1737.4 / obs_m.distance().km)
    
    # 10. 日全食几何判定
    is_bigger = r_moon > r_sun       # 月亮比太阳大
    is_covered = sep < (r_moon + r_sun)  # 发生遮挡
    
    # 只要这几千秒里面，有任何一秒满足全食条件，即返回 True
    return np.any(is_bigger & is_covered)

