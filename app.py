import math
import random
import plotly.graph_objects as go
import streamlit as st

mach_table = {
    1: 336, 2: 332, 3: 328, 4: 324, 5: 320, 6: 316,
    7: 312, 8: 308, 9: 303, 10: 299, 11: 295,
    12: 295, 13: 295, 14: 295, 15: 295, 16: 295, 17: 295
}

PROXY_FUSE_M = 25.0
DT = 0.02
MAX_TIME = 300.0
USE_GRAVITY = True
GRAVITY = 9.81
NOTCH_BREAK_TURN_RATE_DEG = 220.0
NOTCH_HOLD_TURN_RATE_DEG = 85.0
NOTCH_BREAK_TIME = 1.2


def vec_add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def vec_sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def vec_mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def vec_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def vec_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]
def vec_mag(a): return math.sqrt(vec_dot(a, a))
def vec_norm(a):
    m = vec_mag(a)
    if m < 1e-9:
        return [0, 0, 0]
    return [a[0] / m, a[1] / m, a[2] / m]

def direction_3d(horizontal_deg, vertical_deg):
    h = math.radians(horizontal_deg)
    v = math.radians(vertical_deg)
    return [math.cos(v) * math.cos(h), math.cos(v) * math.sin(h), math.sin(v)]

def altitude_drag_factor(alt_km):
    if alt_km <= 0:
        return 1.0
    if alt_km >= 20:
        return 0.25
    return 1.0 - (alt_km / 20.0) * 0.75

def closest_distance_between_steps(prev_target, prev_missile, new_target, new_missile):
    r0 = vec_sub(prev_target, prev_missile)
    r1 = vec_sub(new_target, new_missile)
    dr = vec_sub(r1, r0)
    dr_mag2 = vec_dot(dr, dr)
    if dr_mag2 < 1e-12:
        return vec_mag(r1)
    t = -vec_dot(r0, dr) / dr_mag2
    t = max(0, min(1, t))
    closest = vec_add(r0, vec_mul(dr, t))
    return vec_mag(closest)

def turn_toward_direction(old_dir, desired_dir, turn_rate_deg_per_sec, dt):
    old_dir = vec_norm(old_dir)
    desired_dir = vec_norm(desired_dir)
    dot = max(-1, min(1, vec_dot(old_dir, desired_dir)))
    angle = math.acos(dot)
    if angle < 1e-6:
        return desired_dir
    max_turn = math.radians(turn_rate_deg_per_sec) * dt
    blend = min(1, max_turn / angle)
    return vec_norm([
        old_dir[0] * (1 - blend) + desired_dir[0] * blend,
        old_dir[1] * (1 - blend) + desired_dir[1] * blend,
        old_dir[2] * (1 - blend) + desired_dir[2] * blend,
    ])

def choose_notch_side(target_pos, missile_pos, current_target_dir):
    to_missile = [missile_pos[0] - target_pos[0], missile_pos[1] - target_pos[1], 0]
    los = vec_norm(to_missile)
    if vec_mag(los) < 1e-9:
        return "right"
    right_notch = [los[1], -los[0], 0]
    left_notch = [-los[1], los[0], 0]
    current_horizontal = vec_norm([current_target_dir[0], current_target_dir[1], 0])
    if vec_mag(current_horizontal) < 1e-9:
        current_horizontal = [1, 0, 0]
    return "left" if vec_dot(left_notch, current_horizontal) > vec_dot(right_notch, current_horizontal) else "right"


def run_sim(params):
    target_altitude = params["target_altitude"]
    target_mach_start = params["target_mach_start"]
    target_accel_mach = params["target_accel_mach"]
    target_max_mach = max(params["target_max_mach"], target_mach_start)
    target_heading = params["target_heading"]
    target_climb = params["target_climb"]
    change_target_angle = params["change_target_angle"]
    change_altitude = params["change_altitude"]
    new_target_heading = params["new_target_heading"]
    new_target_climb = params["new_target_climb"]
    notch_mode = params["notch_mode"]
    notch_distance = params["notch_distance"]
    notch_vertical_angle = max(-60, min(60, params["notch_vertical_angle"]))
    missile_altitude = params["missile_altitude"]
    missile_mach_start = params["missile_mach_start"]
    missile_drag_mach = params["missile_drag_mach"]
    start_horizontal_range = params["start_horizontal_range"]
    activation_range = params["activation_range"]
    use_loft = params["use_loft"]
    loft_angle = params["loft_angle"]
    loft_strength = 2.5
    guidance_type = params["guidance_type"]
    nav_constant = params["nav_constant"]
    apn_gain = params["apn_gain"]
    has_lpi = params["has_lpi"]
    lpi_value = params["lpi_value"]
    runs = params["runs"] if has_lpi else 1

    alt_key = max(1, min(17, round(target_altitude)))
    sound_speed = mach_table[alt_key]
    target_max_speed = target_max_mach * sound_speed

    all_hit_times = []
    all_first_ping_distances, all_first_ping_times, all_first_ping_points = [], [], []
    all_activation_times, all_activation_distances, all_activation_points = [], [], []
    all_notch_times, all_notch_points = [], []
    all_angle_change_times, all_angle_change_points = [], []

    final = {}

    for run in range(runs):
        missile_mach = missile_mach_start
        target_mach = target_mach_start
        missile_speed = missile_mach * sound_speed
        target_speed = target_mach * sound_speed
        target_pos = [0, 0, target_altitude * 1000]
        missile_pos = [0, -start_horizontal_range * 1000, missile_altitude * 1000]
        current_target_dir = direction_3d(target_heading, target_climb)
        target_vel = vec_mul(current_target_dir, target_speed)
        missile_dir = vec_norm(vec_sub(target_pos, missile_pos))
        missile_vel = vec_mul(missile_dir, missile_speed)

        time = 0
        ping_timer = 0
        rwr_ping_visible_timer = 0
        reached_activation_range = False
        angle_changed = False
        notch_started = False
        notch_start_time = None
        chosen_notch_side = None
        intercepted = False
        first_ping_distance = None
        first_ping_time = None
        first_ping_point = None

        mx, my, mz, tx, ty, tz = [], [], [], [], [], []
        time_list, missile_mach_list, target_mach_list = [], [], []
        missile_ms_list, target_ms_list, distance_list = [], [], []
        phase_list, target_phase_list, drag_factor_list, actual_drag_list = [], [], [], []

        while time <= MAX_TIME:
            prev_target_pos = target_pos[:]
            prev_missile_pos = missile_pos[:]
            rel_pos = vec_sub(target_pos, missile_pos)
            distance = vec_mag(rel_pos)

            if not reached_activation_range and distance <= activation_range * 1000:
                reached_activation_range = True
                activation_point = (missile_pos[0] / 1000, missile_pos[1] / 1000, missile_pos[2] / 1000)
                all_activation_times.append(time)
                all_activation_distances.append(distance / 1000)
                all_activation_points.append(activation_point)
                if run == 0:
                    final["activation_point"] = activation_point

            if change_target_angle and not angle_changed and not notch_started:
                current_alt_km = target_pos[2] / 1000
                should_change = (target_climb >= 0 and current_alt_km >= change_altitude) or (target_climb < 0 and current_alt_km <= change_altitude)
                if should_change:
                    current_target_dir = direction_3d(new_target_heading, new_target_climb)
                    target_vel = vec_mul(current_target_dir, target_speed)
                    angle_changed = True
                    angle_change_point = (target_pos[0] / 1000, target_pos[1] / 1000, target_pos[2] / 1000)
                    all_angle_change_times.append(time)
                    all_angle_change_points.append(angle_change_point)
                    if run == 0:
                        final["angle_change_point"] = angle_change_point

            if notch_mode != 0 and not notch_started:
                notch_triggered = False
                if notch_mode == 1 and distance <= notch_distance * 1000:
                    notch_triggered = True
                elif notch_mode == 2 and has_lpi and first_ping_distance is not None:
                    notch_triggered = True
                elif notch_mode == 3 and reached_activation_range:
                    notch_triggered = True
                if notch_triggered:
                    notch_started = True
                    notch_start_time = time
                    chosen_notch_side = choose_notch_side(target_pos, missile_pos, current_target_dir)
                    notch_point = (target_pos[0] / 1000, target_pos[1] / 1000, target_pos[2] / 1000)
                    all_notch_times.append(time)
                    all_notch_points.append(notch_point)
                    if run == 0:
                        final["notch_point"] = notch_point
                        final["notch_side"] = chosen_notch_side

            if notch_started:
                should_update_notch = notch_mode in [1, 3] or (notch_mode == 2 and has_lpi and rwr_ping_visible_timer > 0)
                if should_update_notch:
                    to_missile = [missile_pos[0] - target_pos[0], missile_pos[1] - target_pos[1], 0]
                    los = vec_norm(to_missile)
                    if vec_mag(los) < 1e-9:
                        los = [0, -1, 0]
                    if chosen_notch_side == "left":
                        desired_horizontal = [-los[1], los[0], 0]
                    else:
                        desired_horizontal = [los[1], -los[0], 0]
                    v = math.radians(notch_vertical_angle)
                    desired_notch_dir = vec_norm([
                        desired_horizontal[0] * math.cos(v),
                        desired_horizontal[1] * math.cos(v),
                        math.sin(v)
                    ])
                    notch_elapsed = time - notch_start_time
                    turn_rate = NOTCH_BREAK_TURN_RATE_DEG if notch_elapsed < NOTCH_BREAK_TIME else NOTCH_HOLD_TURN_RATE_DEG
                    current_target_dir = turn_toward_direction(current_target_dir, desired_notch_dir, turn_rate, DT)

            target_accel_ms = target_accel_mach * sound_speed
            target_speed = max(0, min(target_speed + target_accel_ms * DT, target_max_speed))
            target_mach = target_speed / sound_speed
            target_vel = vec_mul(current_target_dir, target_speed)
            target_pos = vec_add(target_pos, vec_mul(target_vel, DT))

            rel_pos = vec_sub(target_pos, missile_pos)
            rel_vel = vec_sub(target_vel, missile_vel)
            distance = vec_mag(rel_pos)

            if distance < 1e-9:
                intercepted = True
                break

            if use_loft and not reached_activation_range:
                current_range_km = distance / 1000
                loft_fraction = (current_range_km - activation_range) / (start_horizontal_range - activation_range) if start_horizontal_range > activation_range else 0
                loft_fraction = max(0, min(1, loft_fraction))
                horizontal_distance = math.sqrt((target_pos[0] - missile_pos[0]) ** 2 + (target_pos[1] - missile_pos[1]) ** 2)
                max_loft_offset = min(math.tan(math.radians(loft_angle)) * horizontal_distance, 50000)
                current_loft_offset = max_loft_offset * loft_fraction * loft_strength
                loft_aim_point = [target_pos[0], target_pos[1], target_pos[2] + current_loft_offset]
                desired_dir = vec_norm(vec_sub(loft_aim_point, missile_pos))
                current_loft_angle = math.degrees(math.atan2(loft_aim_point[2] - missile_pos[2], max(horizontal_distance, 1)))
                phase = f"Loft aim {current_loft_angle:.1f}"
                missile_vel = vec_mul(desired_dir, missile_speed)
            else:
                if guidance_type == 1:
                    phase = "Pure Pursuit"
                    desired_dir = vec_norm(rel_pos)
                    missile_vel = vec_mul(desired_dir, missile_speed)
                else:
                    phase = "APN"
                    omega = vec_mul(vec_cross(rel_pos, rel_vel), 1 / max(distance * distance, 1e-9))
                    closing_speed = max(-vec_dot(rel_pos, rel_vel) / max(distance, 1e-9), 0)
                    los_unit = vec_norm(rel_pos)
                    target_accel_vec = vec_mul(current_target_dir, target_accel_ms)
                    pn_accel = vec_mul(vec_cross(los_unit, omega), -nav_constant * closing_speed)
                    target_accel_parallel = vec_mul(los_unit, vec_dot(target_accel_vec, los_unit))
                    target_accel_perp = vec_sub(target_accel_vec, target_accel_parallel)
                    apn_accel = vec_mul(target_accel_perp, apn_gain * nav_constant / 2)
                    commanded_accel = vec_add(pn_accel, apn_accel)
                    missile_vel = vec_add(missile_vel, vec_mul(commanded_accel, DT))

            if USE_GRAVITY:
                missile_vel[2] -= GRAVITY * DT

            missile_speed = vec_mag(missile_vel)
            missile_mach = missile_speed / sound_speed
            missile_alt_km = missile_pos[2] / 1000
            drag_factor = altitude_drag_factor(missile_alt_km)
            actual_drag = missile_drag_mach * drag_factor
            missile_mach = max(0, missile_mach - actual_drag * DT)
            missile_speed = missile_mach * sound_speed
            missile_vel = vec_mul(vec_norm(missile_vel), missile_speed)
            missile_pos = vec_add(missile_pos, vec_mul(missile_vel, DT))

            rel_pos = vec_sub(target_pos, missile_pos)
            distance = vec_mag(rel_pos)
            closest_distance = closest_distance_between_steps(prev_target_pos, prev_missile_pos, target_pos, missile_pos)

            def append_state():
                mx.append(missile_pos[0] / 1000); my.append(missile_pos[1] / 1000); mz.append(missile_pos[2] / 1000)
                tx.append(target_pos[0] / 1000); ty.append(target_pos[1] / 1000); tz.append(target_pos[2] / 1000)
                time_list.append(time)
                missile_mach_list.append(missile_mach); target_mach_list.append(target_mach)
                missile_ms_list.append(missile_speed); target_ms_list.append(target_speed)
                distance_list.append(distance / 1000); phase_list.append(phase)
                if notch_started:
                    notch_elapsed = time - notch_start_time
                    if notch_elapsed < NOTCH_BREAK_TIME:
                        target_phase_list.append("Notching, break turn")
                    elif notch_mode == 2 and has_lpi and rwr_ping_visible_timer <= 0:
                        target_phase_list.append("Notching, holding last correction")
                    else:
                        target_phase_list.append("Notching, hold/update")
                else:
                    target_phase_list.append("Normal")
                drag_factor_list.append(drag_factor); actual_drag_list.append(actual_drag)

            if closest_distance <= PROXY_FUSE_M:
                intercepted = True
                all_hit_times.append(time)
                if first_ping_distance is not None:
                    all_first_ping_distances.append(first_ping_distance)
                    all_first_ping_times.append(first_ping_time)
                    all_first_ping_points.append(first_ping_point)
                append_state()
                if run == 0:
                    final["intercepted"] = True
                    final["end_time"] = time
                    final["end_distance"] = closest_distance / 1000
                break

            if reached_activation_range:
                ping_timer += DT
                if rwr_ping_visible_timer > 0:
                    rwr_ping_visible_timer = max(0, rwr_ping_visible_timer - DT)
                if has_lpi and ping_timer >= 0.5:
                    ping_timer = 0
                    pinged_this_roll = False
                    if distance <= 3000:
                        pinged_this_roll = True
                    elif random.random() < lpi_value:
                        pinged_this_roll = True
                    if pinged_this_roll:
                        rwr_ping_visible_timer = 0.5
                        if first_ping_distance is None:
                            first_ping_distance = distance / 1000
                            first_ping_time = time
                            first_ping_point = (missile_pos[0] / 1000, missile_pos[1] / 1000, missile_pos[2] / 1000)

            append_state()
            time += DT

        if run == 0:
            final.update({
                "mx": mx, "my": my, "mz": mz, "tx": tx, "ty": ty, "tz": tz,
                "time": time_list,
                "missile_mach": missile_mach_list,
                "target_mach": target_mach_list,
                "missile_ms": missile_ms_list,
                "target_ms": target_ms_list,
                "distance": distance_list,
                "phase": phase_list,
                "target_phase": target_phase_list,
                "drag_factor": drag_factor_list,
                "actual_drag": actual_drag_list,
                "ping_point": first_ping_point,
                "ping_distance": first_ping_distance,
                "ping_time": first_ping_time,
            })
            if not intercepted:
                final["intercepted"] = False
                final["end_time"] = time_list[-1] if time_list else time
                final["end_distance"] = distance_list[-1] if distance_list else None

    avg_ping_distance = avg_ping_time = avg_ping_point = None
    if has_lpi and runs > 1 and all_first_ping_distances:
        avg_ping_distance = sum(all_first_ping_distances) / len(all_first_ping_distances)
        avg_ping_time = sum(all_first_ping_times) / len(all_first_ping_times)
        avg_ping_point = (
            sum(p[0] for p in all_first_ping_points) / len(all_first_ping_points),
            sum(p[1] for p in all_first_ping_points) / len(all_first_ping_points),
            sum(p[2] for p in all_first_ping_points) / len(all_first_ping_points),
        )

    return {
        "final": final,
        "runs": runs,
        "has_lpi": has_lpi,
        "hit_times": all_hit_times,
        "activation_times": all_activation_times,
        "activation_distances": all_activation_distances,
        "notch_times": all_notch_times,
        "angle_change_times": all_angle_change_times,
        "avg_ping_distance": avg_ping_distance,
        "avg_ping_time": avg_ping_time,
        "avg_ping_point": avg_ping_point,
        "first_ping_count": len(all_first_ping_distances),
        "params": params,
    }


def build_figure(result):
    final = result["final"]
    params = result["params"]
    if not final.get("mx") or not final.get("tx"):
        return None

    fig = go.Figure()

    missile_hover = []
    for i in range(len(final["mx"])):
        missile_hover.append(
            f"Missile<br>Time: {final['time'][i]:.2f}s<br>Phase: {final['phase'][i]}<br>"
            f"Speed: Mach {final['missile_mach'][i]:.2f}<br>Speed: {final['missile_ms'][i]:.0f} m/s<br>"
            f"Gravity: {'ON' if USE_GRAVITY else 'OFF'}<br>Base drag: {params['missile_drag_mach']:.4f} Mach/sec<br>"
            f"Altitude drag factor: {final['drag_factor'][i]:.2f}<br>Actual drag: {final['actual_drag'][i]:.4f} Mach/sec<br>"
            f"True 3D distance to target: {final['distance'][i]:.2f} km<br>"
            f"X: {final['mx'][i]:.2f} km<br>Y: {final['my'][i]:.2f} km<br>Alt: {final['mz'][i]:.2f} km"
        )

    target_hover = []
    for i in range(len(final["tx"])):
        target_hover.append(
            f"Target<br>Time: {final['time'][i]:.2f}s<br>Phase: {final['target_phase'][i]}<br>"
            f"Speed: Mach {final['target_mach'][i]:.2f}<br>Speed: {final['target_ms'][i]:.0f} m/s<br>"
            f"Max speed: Mach {params['target_max_mach']:.2f}<br>True 3D distance to missile: {final['distance'][i]:.2f} km<br>"
            f"X: {final['tx'][i]:.2f} km<br>Y: {final['ty'][i]:.2f} km<br>Alt: {final['tz'][i]:.2f} km"
        )

    terminal_name = "Pure Pursuit" if params["guidance_type"] == 1 else f"APN, N={params['nav_constant']}"
    guidance_name = f"Loft aim {params['loft_angle']}° + {terminal_name}" if params["use_loft"] else terminal_name

    fig.add_trace(go.Scatter3d(x=final["mx"], y=final["my"], z=final["mz"], mode="lines", name="Missile path", line=dict(width=6), hovertext=missile_hover, hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=final["tx"], y=final["ty"], z=final["tz"], mode="lines", name="Target path", line=dict(width=6), hovertext=target_hover, hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=[final["mx"][0]], y=[final["my"][0]], z=[final["mz"][0]], mode="markers+text", name="Missile start", text=["Missile start"], marker=dict(size=6), hovertext=[f"Missile start<br>Starting horizontal range: {params['start_horizontal_range']:.2f} km<br>Alt: {params['missile_altitude']:.2f} km"], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=[final["tx"][0]], y=[final["ty"][0]], z=[final["tz"][0]], mode="markers+text", name="Target start", text=["Target start"], marker=dict(size=6), hovertext=[f"Target start<br>Mach {params['target_mach_start']:.2f}<br>Max Mach {params['target_max_mach']:.2f}<br>Alt {params['target_altitude']:.2f} km"], hoverinfo="text"))

    if final.get("activation_point") is not None:
        p = final["activation_point"]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Seeker activation range marker", text=["Seeker range"], marker=dict(size=9), hovertext=[f"Seeker activation range marker<br>Activation range: {params['activation_range']:.2f} km"], hoverinfo="text"))

    if final.get("notch_point") is not None:
        p = final["notch_point"]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Target notch start", text=["Notch"], marker=dict(size=9), hovertext=[f"Target started notching<br>Notch mode: {params['notch_mode']}<br>Side: fixed {final.get('notch_side')}<br>Vertical angle: {params['notch_vertical_angle']:.2f}°"], hoverinfo="text"))

    if final.get("angle_change_point") is not None:
        p = final["angle_change_point"]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Target direction change", text=["Target turn"], marker=dict(size=9), hovertext=[f"Target direction change<br>Changed at altitude: {params['change_altitude']:.2f} km<br>New heading: {params['new_target_heading']:.2f}°<br>New climb angle: {params['new_target_climb']:.2f}°"], hoverinfo="text"))

    if final.get("intercepted"):
        end_name = "Intercept"
        end_text = "Intercept"
        end_hover = [f"Intercept<br>Time: {final.get('end_time'):.2f}s<br>Miss distance: {final.get('end_distance') * 1000:.2f} m"]
    else:
        end_name = "Simulation end"
        end_text = "End"
        final_dist_text = f"{final.get('end_distance'):.2f} km" if final.get("end_distance") is not None else "unknown"
        end_hover = [f"Simulation ended without intercept<br>Final distance: {final_dist_text}<br>Time: {final.get('end_time'):.2f}s"]

    fig.add_trace(go.Scatter3d(x=[final["tx"][-1]], y=[final["ty"][-1]], z=[final["tz"][-1]], mode="markers+text", name=end_name, text=[end_text], marker=dict(size=9), hovertext=end_hover, hoverinfo="text"))

    if result["has_lpi"] and result["runs"] == 1 and final.get("ping_point") is not None:
        p = final["ping_point"]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="First ping", text=["First ping"], marker=dict(size=8), hovertext=[f"First ping<br>Distance: {final['ping_distance']:.2f} km<br>Time: {final['ping_time']:.2f}s"], hoverinfo="text"))

    if result["has_lpi"] and result["runs"] > 1 and result["avg_ping_point"] is not None:
        p = result["avg_ping_point"]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Average first ping", text=["Avg ping"], marker=dict(size=10), hovertext=[f"Average first ping<br>Distance: {result['avg_ping_distance']:.2f} km<br>Time: {result['avg_ping_time']:.2f}s<br>Based on {result['first_ping_count']} ping events"], hoverinfo="text"))

    fig.update_layout(title=f"3D Missile Intercept - {guidance_name}", scene=dict(xaxis_title="X km", yaxis_title="Y km", zaxis_title="Altitude km", aspectmode="data"), width=1000, height=750)
    return fig


st.set_page_config(page_title="3D Missile Intercept Simulator", layout="wide")
st.title("3D Missile Intercept Simulator")

with st.sidebar:
    st.header("Target")
    target_altitude = st.number_input("Target altitude km", value=12.0, step=1.0)
    target_mach_start = st.number_input("Target speed Mach", value=1.0, step=0.1)
    target_accel_mach = st.number_input("Target acceleration Mach/sec", value=0.0, step=0.01, format="%.3f")
    target_max_mach = st.number_input("Target maximum speed Mach", value=float(target_mach_start), step=0.1)
    target_heading = st.number_input("Target horizontal heading degrees", value=0.0, step=5.0)
    target_climb = st.number_input("Target climb/descent angle degrees", value=0.0, step=1.0)

    change_target_angle = st.checkbox("Change target direction after altitude")
    if change_target_angle:
        change_altitude = st.number_input("Altitude where target changes direction km", value=float(target_altitude), step=1.0)
        new_target_heading = st.number_input("New target horizontal heading degrees", value=float(target_heading), step=5.0)
        new_target_climb = st.number_input("New target climb/descent angle degrees", value=float(target_climb), step=1.0)
    else:
        change_altitude = target_altitude
        new_target_heading = target_heading
        new_target_climb = target_climb

    st.header("Notching")
    notch_mode_label = st.selectbox("Target notch mode", ["0 = No notch", "1 = Notch at chosen missile distance", "2 = Notch after first RWR ping, LPI only", "3 = Notch at seeker activation range"])
    notch_mode = int(notch_mode_label[0])
    notch_distance = 8.0
    if notch_mode == 1:
        notch_distance = st.number_input("Start notching when missile is this far away km", value=8.0, step=1.0)
    notch_vertical_angle = 0.0
    if notch_mode != 0:
        notch_vertical_angle = st.number_input("Target vertical angle while notching degrees", value=0.0, step=1.0)

    st.header("Missile / Launch")
    missile_altitude = st.number_input("Missile / launch aircraft altitude km", value=float(target_altitude), step=1.0)
    missile_mach_start = st.number_input("Missile speed Mach", value=4.0, step=0.1)
    missile_drag_mach = st.number_input("Base missile drag Mach/sec at sea level", value=0.02, step=0.005, format="%.4f")
    start_horizontal_range = st.number_input("Starting horizontal distance from target km", value=40.0, step=1.0)
    activation_range = st.number_input("Seeker activation range km", value=12.0, step=1.0)

    st.header("Lofting")
    use_loft = st.checkbox("Use lofting before seeker range")
    loft_angle = 0.0
    if use_loft:
        loft_angle = st.number_input("Maximum missile loft angle degrees", value=25.0, step=1.0)

    st.header("Guidance")
    guidance_label = st.selectbox("Terminal guidance type", ["1 = Pure Pursuit", "2 = APN"])
    guidance_type = int(guidance_label[0])
    nav_constant = 4.0
    apn_gain = 1.0
    if guidance_type == 2:
        nav_constant = st.number_input("Navigation constant N", value=4.0, step=0.5)
        apn_gain = st.number_input("APN target acceleration gain", value=1.0, step=0.1)

    st.header("LPI / Runs")
    has_lpi = st.checkbox("Missile has LPI")
    lpi_value = 0.0
    runs = 1
    if has_lpi:
        lpi_value = st.number_input("LPI value", value=0.07, step=0.01, format="%.3f")
        runs = st.number_input("Simulation runs", min_value=1, max_value=200, value=1, step=1)

    run_button = st.button("Run simulation", type="primary")

params = dict(
    target_altitude=target_altitude,
    target_mach_start=target_mach_start,
    target_accel_mach=target_accel_mach,
    target_max_mach=target_max_mach,
    target_heading=target_heading,
    target_climb=target_climb,
    change_target_angle=change_target_angle,
    change_altitude=change_altitude,
    new_target_heading=new_target_heading,
    new_target_climb=new_target_climb,
    notch_mode=notch_mode,
    notch_distance=notch_distance,
    notch_vertical_angle=notch_vertical_angle,
    missile_altitude=missile_altitude,
    missile_mach_start=missile_mach_start,
    missile_drag_mach=missile_drag_mach,
    start_horizontal_range=start_horizontal_range,
    activation_range=activation_range,
    use_loft=use_loft,
    loft_angle=loft_angle,
    guidance_type=guidance_type,
    nav_constant=nav_constant,
    apn_gain=apn_gain,
    has_lpi=has_lpi,
    lpi_value=lpi_value,
    runs=int(runs),
)

if run_button:
    with st.spinner("Running simulation..."):
        result = run_sim(params)
        fig = build_figure(result)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if result["hit_times"]:
            st.metric("Avg intercept time", f"{sum(result['hit_times']) / len(result['hit_times']):.2f}s")
        else:
            st.metric("Intercept", "Failed")
    with c2:
        if result["activation_times"]:
            st.metric("Seeker range reached", f"{sum(result['activation_distances']) / len(result['activation_distances']):.2f} km")
        else:
            st.metric("Seeker range", "Not reached")
    with c3:
        if params["notch_mode"] != 0 and result["notch_times"]:
            st.metric("Notch start", f"{sum(result['notch_times']) / len(result['notch_times']):.2f}s")
        else:
            st.metric("Notch", "None")
    with c4:
        if result["has_lpi"] and result["runs"] > 1 and result["avg_ping_distance"] is not None:
            st.metric("Avg first ping", f"{result['avg_ping_distance']:.2f} km")
        elif result["has_lpi"] and result["runs"] == 1 and result["final"].get("ping_distance") is not None:
            st.metric("First ping", f"{result['final']['ping_distance']:.2f} km")
        else:
            st.metric("Ping", "None")

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Set the values in the sidebar, then click Run simulation.")
