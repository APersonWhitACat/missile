import math
import random
import streamlit as st
import plotly.graph_objects as go

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

NOTCH_BREAK_TIME = 1.2


def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec_mul(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]


def vec_mag(a):
    return math.sqrt(vec_dot(a, a))


def vec_norm(a):
    m = vec_mag(a)
    if m < 1e-9:
        return [0, 0, 0]
    return [a[0] / m, a[1] / m, a[2] / m]


def horizontal_norm(v):
    h = [v[0], v[1], 0]
    return vec_norm(h)


def relative_direction_from_line(base_forward, horizontal_relative_deg, vertical_deg):
    forward = horizontal_norm(base_forward)

    if vec_mag(forward) < 1e-9:
        forward = [0, 1, 0]

    right = [forward[1], -forward[0], 0]

    h = math.radians(horizontal_relative_deg)
    v = math.radians(vertical_deg)

    horizontal_dir = vec_add(
        vec_mul(forward, math.cos(h)),
        vec_mul(right, math.sin(h))
    )

    horizontal_dir = vec_norm(horizontal_dir)

    return vec_norm([
        horizontal_dir[0] * math.cos(v),
        horizontal_dir[1] * math.cos(v),
        math.sin(v)
    ])


def target_relative_direction(target_pos, missile_pos, heading_relative_deg, vertical_deg):
    away_from_missile = vec_sub(target_pos, missile_pos)

    return relative_direction_from_line(
        away_from_missile,
        heading_relative_deg,
        vertical_deg
    )


def manual_launch_direction(missile_pos, target_pos, horizontal_offset_deg, vertical_angle_deg):
    toward_target = vec_sub(target_pos, missile_pos)

    return relative_direction_from_line(
        toward_target,
        horizontal_offset_deg,
        vertical_angle_deg
    )


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

    new_dir = [
        old_dir[0] * (1 - blend) + desired_dir[0] * blend,
        old_dir[1] * (1 - blend) + desired_dir[1] * blend,
        old_dir[2] * (1 - blend) + desired_dir[2] * blend
    ]

    return vec_norm(new_dir)


def limit_missile_g(old_vel, commanded_vel, max_g, dt):
    old_speed = vec_mag(old_vel)
    commanded_speed = vec_mag(commanded_vel)

    if old_speed < 1e-6 or commanded_speed < 1e-6:
        return commanded_vel

    old_dir = vec_norm(old_vel)
    commanded_dir = vec_norm(commanded_vel)

    max_lat_accel = max_g * GRAVITY
    max_turn_rad = (max_lat_accel / max(old_speed, 1.0)) * dt

    dot = max(-1, min(1, vec_dot(old_dir, commanded_dir)))
    angle = math.acos(dot)

    if angle <= max_turn_rad:
        return commanded_vel

    blend = max_turn_rad / max(angle, 1e-9)

    limited_dir = vec_norm([
        old_dir[0] * (1 - blend) + commanded_dir[0] * blend,
        old_dir[1] * (1 - blend) + commanded_dir[1] * blend,
        old_dir[2] * (1 - blend) + commanded_dir[2] * blend
    ])

    return vec_mul(limited_dir, commanded_speed)


def motor_thrust_at_time(time_s, booster_thrust, booster_time, sustainer_thrust, sustainer_time):
    if time_s < booster_time:
        return booster_thrust

    if time_s < booster_time + sustainer_time:
        return sustainer_thrust

    return 0.0


def choose_notch_side(target_pos, missile_pos, current_target_dir):
    to_missile = [
        missile_pos[0] - target_pos[0],
        missile_pos[1] - target_pos[1],
        0
    ]

    los = vec_norm(to_missile)

    if vec_mag(los) < 1e-9:
        return "right"

    right_notch = [los[1], -los[0], 0]
    left_notch = [-los[1], los[0], 0]

    current_horizontal = [current_target_dir[0], current_target_dir[1], 0]
    current_horizontal = vec_norm(current_horizontal)

    if vec_mag(current_horizontal) < 1e-9:
        current_horizontal = [1, 0, 0]

    if vec_dot(left_notch, current_horizontal) > vec_dot(right_notch, current_horizontal):
        return "left"

    return "right"


def lpi_detection_chance(distance_m, base_lpi):
    if distance_m <= 3000:
        return 1.0

    if distance_m >= 8000:
        return base_lpi

    closeness = (8000 - distance_m) / 5000
    closeness = max(0, min(1, closeness))

    return base_lpi + closeness * (1.0 - base_lpi)


def make_animation_figure(result, frame_skip, play_speed_ms):
    final_mx = result["final_mx"]
    final_my = result["final_my"]
    final_mz = result["final_mz"]
    final_tx = result["final_tx"]
    final_ty = result["final_ty"]
    final_tz = result["final_tz"]
    final_time = result["final_time"]

    if not final_mx or not final_tx:
        return None

    frame_skip = max(1, int(frame_skip))
    play_speed_ms = max(5, int(play_speed_ms))

    indices = list(range(0, len(final_mx), frame_skip))
    if indices[-1] != len(final_mx) - 1:
        indices.append(len(final_mx) - 1)

    all_x = final_mx + final_tx
    all_y = final_my + final_ty
    all_z = final_mz + final_tz

    pad_x = max(1.0, (max(all_x) - min(all_x)) * 0.08)
    pad_y = max(1.0, (max(all_y) - min(all_y)) * 0.08)
    pad_z = max(1.0, (max(all_z) - min(all_z)) * 0.08)

    x_range = [min(all_x) - pad_x, max(all_x) + pad_x]
    y_range = [min(all_y) - pad_y, max(all_y) + pad_y]
    z_range = [min(all_z) - pad_z, max(all_z) + pad_z]

    start_i = indices[0]

    fig_anim = go.Figure(
        data=[
            go.Scatter3d(
                x=final_mx[:start_i + 1],
                y=final_my[:start_i + 1],
                z=final_mz[:start_i + 1],
                mode="lines",
                name="Missile trail",
                line=dict(width=6)
            ),
            go.Scatter3d(
                x=final_tx[:start_i + 1],
                y=final_ty[:start_i + 1],
                z=final_tz[:start_i + 1],
                mode="lines",
                name="Target trail",
                line=dict(width=6)
            ),
            go.Scatter3d(
                x=[final_mx[start_i]],
                y=[final_my[start_i]],
                z=[final_mz[start_i]],
                mode="markers+text",
                name="Missile",
                text=["Missile"],
                marker=dict(size=6),
                hovertext=[f"Missile<br>Time: {final_time[start_i]:.2f}s"],
                hoverinfo="text"
            ),
            go.Scatter3d(
                x=[final_tx[start_i]],
                y=[final_ty[start_i]],
                z=[final_tz[start_i]],
                mode="markers+text",
                name="Target",
                text=["Target"],
                marker=dict(size=6),
                hovertext=[f"Target<br>Time: {final_time[start_i]:.2f}s"],
                hoverinfo="text"
            ),
        ]
    )

    frames = []

    for i in indices:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=final_mx[:i + 1],
                        y=final_my[:i + 1],
                        z=final_mz[:i + 1],
                        mode="lines",
                        name="Missile trail",
                        line=dict(width=6)
                    ),
                    go.Scatter3d(
                        x=final_tx[:i + 1],
                        y=final_ty[:i + 1],
                        z=final_tz[:i + 1],
                        mode="lines",
                        name="Target trail",
                        line=dict(width=6)
                    ),
                    go.Scatter3d(
                        x=[final_mx[i]],
                        y=[final_my[i]],
                        z=[final_mz[i]],
                        mode="markers+text",
                        name="Missile",
                        text=["Missile"],
                        marker=dict(size=6),
                        hovertext=[f"Missile<br>Time: {final_time[i]:.2f}s"],
                        hoverinfo="text"
                    ),
                    go.Scatter3d(
                        x=[final_tx[i]],
                        y=[final_ty[i]],
                        z=[final_tz[i]],
                        mode="markers+text",
                        name="Target",
                        text=["Target"],
                        marker=dict(size=6),
                        hovertext=[f"Target<br>Time: {final_time[i]:.2f}s"],
                        hoverinfo="text"
                    ),
                ],
                name=str(i)
            )
        )

    fig_anim.frames = frames

    slider_steps = []

    for i in indices:
        slider_steps.append(
            {
                "args": [
                    [str(i)],
                    {
                        "frame": {"duration": play_speed_ms, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ],
                "label": f"{final_time[i]:.1f}s",
                "method": "animate"
            }
        )

    fig_anim.update_layout(
        title="Animated Playback",
        scene=dict(
            xaxis_title="X km",
            yaxis_title="Y km",
            zaxis_title="Altitude km",
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode="data"
        ),
        height=750,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 0,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": play_speed_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": "immediate"
                            }
                        ]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ]
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Sim time: "},
                "pad": {"t": 50},
                "steps": slider_steps
            }
        ]
    )

    return fig_anim


st.set_page_config(page_title="3D Missile Intercept Simulator", layout="wide")
st.title("3D Missile Intercept Simulator")

with st.sidebar:
    st.header("Target Properties")
    target_altitude = st.number_input("Target altitude km", value=12.0, step=0.5)
    target_mach_start = st.number_input("Target speed Mach", value=1.0, step=0.1)
    target_accel_mach = st.number_input("Target acceleration Mach/sec", value=0.0, step=0.01)
    target_max_mach = st.number_input("Target maximum speed Mach", value=float(target_mach_start), step=0.1)

    target_heading = st.number_input(
        "Target heading relative to missile degrees, 0=away, 180=toward, 90=right, -90=left",
        value=0.0,
        step=5.0
    )

    target_climb = st.number_input(
        "Target vertical angle degrees, +up / -down",
        value=0.0,
        step=1.0
    )

    target_turn_rate_deg = st.number_input(
        "Target turn rate deg/sec",
        value=45.0,
        min_value=1.0,
        step=1.0
    )

    if target_max_mach < target_mach_start:
        target_max_mach = target_mach_start
        st.warning("Target max Mach was raised to match starting Mach.")

    change_target_angle = st.checkbox("Change target direction after reaching altitude", value=False)

    change_altitude = None
    new_target_heading = None
    new_target_climb = None

    if change_target_angle:
        change_altitude = st.number_input(
            "Altitude where target changes direction km",
            value=float(target_altitude),
            step=0.5
        )

        new_target_heading = st.number_input(
            "New target heading relative to missile degrees, 0=away, 180=toward, 90=right, -90=left",
            value=float(target_heading),
            step=5.0
        )

        new_target_climb = st.number_input(
            "New target vertical angle degrees, +up / -down",
            value=float(target_climb),
            step=1.0
        )

    st.header("Target Notching")
    notch_mode = st.selectbox(
        "Target notch mode",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 = No notch",
            1: "1 = Notch at chosen missile distance",
            2: "2 = Notch after first RWR ping, LPI only",
            3: "3 = Notch at seeker activation range"
        }[x]
    )

    notch_distance = None
    notch_vertical_angle = 0.0

    if notch_mode != 0:
        if notch_mode == 1:
            notch_distance = st.number_input(
                "Start notching when missile is this far away km",
                value=8.0,
                step=0.5
            )

        notch_vertical_angle = st.number_input(
            "Target vertical angle while notching degrees, +up / -down",
            value=0.0,
            step=1.0
        )
        notch_vertical_angle = max(-60, min(60, notch_vertical_angle))

    st.header("Missile / Launch Platform")
    missile_altitude = st.number_input("Launch platform altitude km", value=float(target_altitude), step=0.5)
    launch_platform_mach = st.number_input("Launch platform speed Mach", value=1.2, step=0.1)

    use_manual_launch = st.checkbox("Use manual launch direction", value=False)

    launch_horizontal_offset_deg = 0.0
    launch_vertical_angle_deg = 0.0

    if use_manual_launch:
        launch_horizontal_offset_deg = st.number_input(
            "Launch horizontal offset degrees, 0=toward target, 90=right, -90=left",
            value=0.0,
            step=5.0
        )

        launch_vertical_angle_deg = st.number_input(
            "Launch vertical angle degrees, +up / -down",
            value=20.0,
            step=5.0
        )

    missile_mass_kg = st.number_input("Missile mass kg", value=168.0, min_value=1.0, step=1.0)

    booster_thrust_n = st.number_input("Booster thrust N", value=19500.0, min_value=0.0, step=500.0)
    booster_burn_time = st.number_input("Booster burn time s", value=8.0, min_value=0.0, step=0.5)

    sustainer_thrust_n = st.number_input("Sustainer thrust N", value=0.0, min_value=0.0, step=500.0)
    sustainer_burn_time = st.number_input("Sustainer burn time s", value=0.0, min_value=0.0, step=0.5)

    missile_max_g = st.number_input("Missile max G", value=40.0, min_value=1.0, step=1.0)

    missile_drag_mach = st.number_input(
        "Base missile drag Mach/sec at sea level",
        value=0.02,
        step=0.005,
        format="%.4f"
    )

    start_horizontal_range = st.number_input("Starting horizontal distance from target km", value=40.0, step=1.0)
    activation_range = st.number_input("Seeker activation range km", value=12.0, step=0.5)

    st.header("Lofting")
    use_loft = st.checkbox("Use automatic lofting before seeker range", value=False)

    loft_angle = 0.0
    loft_strength = 2.5

    if use_loft:
        loft_angle = st.number_input("Maximum automatic loft angle degrees", value=25.0, step=1.0)

    st.header("Terminal Guidance")
    st.write("Guidance: document-style APN")

    nav_constant = st.number_input("Navigation constant N", value=4.0, step=0.1)
    apn_gain = st.number_input("APN target acceleration gain", value=1.0, step=0.1)

    st.header("LPI / Simulation")
    has_lpi = st.checkbox("Does missile have LPI", value=False)

    lpi_value = 0.0
    runs = 1

    if has_lpi:
        lpi_value = st.number_input(
            "Base LPI value",
            value=0.07,
            min_value=0.0,
            max_value=1.0,
            step=0.01
        )
        runs = st.number_input("Simulation runs", value=1, min_value=1, max_value=200, step=1)
    else:
        runs = 1

    st.header("Playback")
    show_animation = st.checkbox("Show animated playback", value=False)

    animation_frame_skip = 20
    animation_speed_ms = 60

    if show_animation:
        animation_speed_ms = st.number_input(
            "Playback speed ms per frame",
            value=60,
            min_value=5,
            max_value=1000,
            step=5
        )

        animation_frame_skip = st.number_input(
            "Animation frame skip",
            value=20,
            min_value=1,
            max_value=200,
            step=1
        )

    run_button = st.button("Run simulation", type="primary")


def run_simulation():
    alt_key = round(target_altitude)
    alt_key = max(1, min(17, alt_key))
    sound_speed = mach_table[alt_key]

    target_max_speed = target_max_mach * sound_speed

    all_hit_times = []
    all_activation_to_hit_times = []

    all_first_ping_distances = []
    all_first_ping_times = []
    all_first_ping_points = []

    all_activation_times = []
    all_activation_distances = []

    all_notch_times = []
    all_angle_change_times = []

    final_mx, final_my, final_mz = [], [], []
    final_tx, final_ty, final_tz = [], [], []

    final_time = []
    final_missile_mach = []
    final_target_mach = []
    final_missile_ms = []
    final_target_ms = []
    final_distance = []
    final_phase = []
    final_target_phase = []
    final_drag_factor = []
    final_actual_drag = []
    final_turn_rate = []
    final_thrust = []
    final_motor_accel = []
    final_target_accel = []
    final_target_accel_perp = []

    final_ping_point = None
    final_ping_distance = None
    final_ping_time = None
    final_activation_point = None
    final_activation_time = None
    final_notch_point = None
    final_angle_change_point = None
    final_notch_side = None

    final_intercepted = False
    final_end_time = None
    final_end_distance = None
    final_activation_to_intercept_time = None

    for run in range(int(runs)):
        missile_mach = launch_platform_mach
        target_mach = target_mach_start

        missile_speed = missile_mach * sound_speed
        target_speed = target_mach * sound_speed

        target_pos = [0, 0, target_altitude * 1000]
        missile_pos = [0, -start_horizontal_range * 1000, missile_altitude * 1000]

        current_target_dir = target_relative_direction(
            target_pos,
            missile_pos,
            target_heading,
            target_climb
        )

        target_vel = vec_mul(current_target_dir, target_speed)

        if use_manual_launch:
            missile_dir = manual_launch_direction(
                missile_pos,
                target_pos,
                launch_horizontal_offset_deg,
                launch_vertical_angle_deg
            )
        else:
            missile_dir = vec_norm(vec_sub(target_pos, missile_pos))

        missile_vel = vec_mul(missile_dir, missile_speed)

        time = 0
        activation_time = None
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

        mx, my, mz = [], [], []
        tx, ty, tz = [], [], []

        time_list = []
        missile_mach_list = []
        target_mach_list = []
        missile_ms_list = []
        target_ms_list = []
        distance_list = []
        phase_list = []
        target_phase_list = []
        drag_factor_list = []
        actual_drag_list = []
        turn_rate_list = []
        thrust_list = []
        motor_accel_list = []
        target_accel_list = []
        target_accel_perp_list = []

        while time <= MAX_TIME:
            prev_target_pos = target_pos[:]
            prev_missile_pos = missile_pos[:]
            prev_missile_vel = missile_vel[:]
            prev_target_vel = target_vel[:]

            rel_pos = vec_sub(target_pos, missile_pos)
            distance = vec_mag(rel_pos)

            if not reached_activation_range and distance <= activation_range * 1000:
                reached_activation_range = True
                activation_time = time

                activation_point = (
                    missile_pos[0] / 1000,
                    missile_pos[1] / 1000,
                    missile_pos[2] / 1000
                )

                all_activation_times.append(time)
                all_activation_distances.append(distance / 1000)

                if run == 0:
                    final_activation_point = activation_point
                    final_activation_time = activation_time

            if change_target_angle and not angle_changed and not notch_started:
                current_alt_km = target_pos[2] / 1000

                should_change = False

                if target_climb >= 0 and current_alt_km >= change_altitude:
                    should_change = True
                elif target_climb < 0 and current_alt_km <= change_altitude:
                    should_change = True

                if should_change:
                    desired_target_dir = target_relative_direction(
                        target_pos,
                        missile_pos,
                        new_target_heading,
                        new_target_climb
                    )

                    current_target_dir = turn_toward_direction(
                        current_target_dir,
                        desired_target_dir,
                        target_turn_rate_deg,
                        DT
                    )

                    target_vel = vec_mul(current_target_dir, target_speed)

                    if vec_dot(current_target_dir, desired_target_dir) > 0.999:
                        angle_changed = True

                    angle_change_point = (
                        target_pos[0] / 1000,
                        target_pos[1] / 1000,
                        target_pos[2] / 1000
                    )

                    if len(all_angle_change_times) == 0 or run == 0:
                        all_angle_change_times.append(time)

                    if run == 0 and final_angle_change_point is None:
                        final_angle_change_point = angle_change_point

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

                    chosen_notch_side = choose_notch_side(
                        target_pos,
                        missile_pos,
                        current_target_dir
                    )

                    notch_point = (
                        target_pos[0] / 1000,
                        target_pos[1] / 1000,
                        target_pos[2] / 1000
                    )

                    all_notch_times.append(time)

                    if run == 0:
                        final_notch_point = notch_point
                        final_notch_side = chosen_notch_side

            if notch_started:
                should_update_notch = False

                if notch_mode in [1, 3]:
                    should_update_notch = True
                elif notch_mode == 2 and has_lpi:
                    if rwr_ping_visible_timer > 0:
                        should_update_notch = True

                if should_update_notch:
                    to_missile = [
                        missile_pos[0] - target_pos[0],
                        missile_pos[1] - target_pos[1],
                        0
                    ]

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

                    notch_break_rate = target_turn_rate_deg * 2.0
                    notch_hold_rate = target_turn_rate_deg

                    if notch_elapsed < NOTCH_BREAK_TIME:
                        current_target_dir = turn_toward_direction(
                            current_target_dir,
                            desired_notch_dir,
                            notch_break_rate,
                            DT
                        )
                    else:
                        current_target_dir = turn_toward_direction(
                            current_target_dir,
                            desired_notch_dir,
                            notch_hold_rate,
                            DT
                        )

            target_accel_ms = target_accel_mach * sound_speed
            target_speed = target_speed + target_accel_ms * DT
            target_speed = max(0, min(target_speed, target_max_speed))
            target_mach = target_speed / sound_speed

            target_vel = vec_mul(current_target_dir, target_speed)
            target_pos = vec_add(target_pos, vec_mul(target_vel, DT))

            actual_target_accel_vec = vec_mul(
                vec_sub(target_vel, prev_target_vel),
                1 / max(DT, 1e-9)
            )

            rel_pos = vec_sub(target_pos, missile_pos)
            rel_vel = vec_sub(target_vel, missile_vel)
            distance = vec_mag(rel_pos)

            if distance < 1e-9:
                intercepted = True
                break

            if use_loft and not reached_activation_range:
                current_range_km = distance / 1000

                if start_horizontal_range > activation_range:
                    loft_fraction = (current_range_km - activation_range) / (start_horizontal_range - activation_range)
                else:
                    loft_fraction = 0

                loft_fraction = max(0, min(1, loft_fraction))

                horizontal_distance = math.sqrt(
                    (target_pos[0] - missile_pos[0]) ** 2 +
                    (target_pos[1] - missile_pos[1]) ** 2
                )

                max_loft_offset = math.tan(math.radians(loft_angle)) * horizontal_distance
                max_loft_offset = min(max_loft_offset, 50000)

                current_loft_offset = max_loft_offset * loft_fraction * loft_strength

                loft_aim_point = [
                    target_pos[0],
                    target_pos[1],
                    target_pos[2] + current_loft_offset
                ]

                desired_dir = vec_norm(vec_sub(loft_aim_point, missile_pos))

                current_loft_angle = math.degrees(math.atan2(
                    loft_aim_point[2] - missile_pos[2],
                    max(horizontal_distance, 1)
                ))

                phase = f"Auto loft aim {current_loft_angle:.1f}"
                commanded_missile_vel = vec_mul(desired_dir, missile_speed)

                target_accel_perp = [0, 0, 0]
                target_accel_perp_mag = 0.0

            else:
                phase = "Doc-style APN"

                omega = vec_mul(
                    vec_cross(rel_pos, rel_vel),
                    1 / max(distance * distance, 1e-9)
                )

                closing_speed = -vec_dot(rel_pos, rel_vel) / max(distance, 1e-9)
                closing_speed = max(closing_speed, 0)

                los_unit = vec_norm(rel_pos)

                pn_accel = vec_mul(
                    vec_cross(los_unit, omega),
                    -nav_constant * closing_speed
                )

                target_accel_parallel = vec_mul(
                    los_unit,
                    vec_dot(actual_target_accel_vec, los_unit)
                )

                target_accel_perp = vec_sub(
                    actual_target_accel_vec,
                    target_accel_parallel
                )

                target_accel_perp_mag = vec_mag(target_accel_perp)

                apn_accel = vec_mul(
                    target_accel_perp,
                    apn_gain * nav_constant / 2
                )

                commanded_accel = vec_add(pn_accel, apn_accel)

                commanded_missile_vel = vec_add(
                    missile_vel,
                    vec_mul(commanded_accel, DT)
                )

            missile_vel = limit_missile_g(
                missile_vel,
                commanded_missile_vel,
                missile_max_g,
                DT
            )

            current_thrust_n = motor_thrust_at_time(
                time,
                booster_thrust_n,
                booster_burn_time,
                sustainer_thrust_n,
                sustainer_burn_time
            )

            motor_accel_ms2 = current_thrust_n / max(missile_mass_kg, 1.0)

            if vec_mag(missile_vel) > 1e-6:
                thrust_dir = vec_norm(missile_vel)
            else:
                thrust_dir = vec_norm(rel_pos)

            missile_vel = vec_add(
                missile_vel,
                vec_mul(thrust_dir, motor_accel_ms2 * DT)
            )

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

            old_dir_for_turn = vec_norm(prev_missile_vel)
            new_dir_for_turn = vec_norm(missile_vel)
            turn_dot = max(-1, min(1, vec_dot(old_dir_for_turn, new_dir_for_turn)))
            turn_angle = math.degrees(math.acos(turn_dot))
            actual_turn_rate = turn_angle / DT if DT > 0 else 0

            rel_pos = vec_sub(target_pos, missile_pos)
            distance = vec_mag(rel_pos)

            closest_distance = closest_distance_between_steps(
                prev_target_pos,
                prev_missile_pos,
                target_pos,
                missile_pos
            )

            if closest_distance <= PROXY_FUSE_M:
                intercepted = True
                all_hit_times.append(time)

                if activation_time is not None:
                    all_activation_to_hit_times.append(time - activation_time)

                mx.append(missile_pos[0] / 1000)
                my.append(missile_pos[1] / 1000)
                mz.append(missile_pos[2] / 1000)

                tx.append(target_pos[0] / 1000)
                ty.append(target_pos[1] / 1000)
                tz.append(target_pos[2] / 1000)

                time_list.append(time)
                missile_mach_list.append(missile_mach)
                target_mach_list.append(target_mach)
                missile_ms_list.append(missile_speed)
                target_ms_list.append(target_speed)
                distance_list.append(distance / 1000)
                phase_list.append(phase)

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

                drag_factor_list.append(drag_factor)
                actual_drag_list.append(actual_drag)
                turn_rate_list.append(actual_turn_rate)
                thrust_list.append(current_thrust_n)
                motor_accel_list.append(motor_accel_ms2)
                target_accel_list.append(vec_mag(actual_target_accel_vec))
                target_accel_perp_list.append(target_accel_perp_mag)

                if run == 0:
                    final_intercepted = True
                    final_end_time = time
                    final_end_distance = closest_distance / 1000

                    if activation_time is not None:
                        final_activation_to_intercept_time = time - activation_time

                break

            if reached_activation_range:
                ping_timer += DT

                if rwr_ping_visible_timer > 0:
                    rwr_ping_visible_timer = max(0, rwr_ping_visible_timer - DT)

                if has_lpi and ping_timer >= 0.5:
                    ping_timer = 0

                    ping_chance = lpi_detection_chance(distance, lpi_value)

                    if random.random() < ping_chance:
                        rwr_ping_visible_timer = 0.5

                        if first_ping_distance is None:
                            first_ping_distance = distance / 1000
                            first_ping_time = time
                            first_ping_point = (
                                missile_pos[0] / 1000,
                                missile_pos[1] / 1000,
                                missile_pos[2] / 1000
                            )

            mx.append(missile_pos[0] / 1000)
            my.append(missile_pos[1] / 1000)
            mz.append(missile_pos[2] / 1000)

            tx.append(target_pos[0] / 1000)
            ty.append(target_pos[1] / 1000)
            tz.append(target_pos[2] / 1000)

            time_list.append(time)
            missile_mach_list.append(missile_mach)
            target_mach_list.append(target_mach)
            missile_ms_list.append(missile_speed)
            target_ms_list.append(target_speed)
            distance_list.append(distance / 1000)
            phase_list.append(phase)

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

            drag_factor_list.append(drag_factor)
            actual_drag_list.append(actual_drag)
            turn_rate_list.append(actual_turn_rate)
            thrust_list.append(current_thrust_n)
            motor_accel_list.append(motor_accel_ms2)
            target_accel_list.append(vec_mag(actual_target_accel_vec))
            target_accel_perp_list.append(target_accel_perp_mag)

            time += DT

        if first_ping_distance is not None:
            all_first_ping_distances.append(first_ping_distance)
            all_first_ping_times.append(first_ping_time)
            all_first_ping_points.append(first_ping_point)

        if run == 0:
            final_mx, final_my, final_mz = mx, my, mz
            final_tx, final_ty, final_tz = tx, ty, tz

            final_time = time_list
            final_missile_mach = missile_mach_list
            final_target_mach = target_mach_list
            final_missile_ms = missile_ms_list
            final_target_ms = target_ms_list
            final_distance = distance_list
            final_phase = phase_list
            final_target_phase = target_phase_list
            final_drag_factor = drag_factor_list
            final_actual_drag = actual_drag_list
            final_turn_rate = turn_rate_list
            final_thrust = thrust_list
            final_motor_accel = motor_accel_list
            final_target_accel = target_accel_list
            final_target_accel_perp = target_accel_perp_list

            final_ping_point = first_ping_point
            final_ping_distance = first_ping_distance
            final_ping_time = first_ping_time

            if not intercepted:
                final_intercepted = False
                if distance_list:
                    final_end_time = time_list[-1]
                    final_end_distance = distance_list[-1]
                else:
                    final_end_time = time
                    final_end_distance = None

    avg_ping_distance = None
    avg_ping_time = None
    avg_ping_point = None

    if has_lpi and all_first_ping_distances and runs > 1:
        avg_ping_distance = sum(all_first_ping_distances) / len(all_first_ping_distances)
        avg_ping_time = sum(all_first_ping_times) / len(all_first_ping_times)

        avg_ping_x = sum(p[0] for p in all_first_ping_points) / len(all_first_ping_points)
        avg_ping_y = sum(p[1] for p in all_first_ping_points) / len(all_first_ping_points)
        avg_ping_z = sum(p[2] for p in all_first_ping_points) / len(all_first_ping_points)
        avg_ping_point = (avg_ping_x, avg_ping_y, avg_ping_z)

    avg_activation_to_hit_time = None

    if all_activation_to_hit_times:
        avg_activation_to_hit_time = sum(all_activation_to_hit_times) / len(all_activation_to_hit_times)

    return {
        "all_hit_times": all_hit_times,
        "all_activation_to_hit_times": all_activation_to_hit_times,
        "avg_activation_to_hit_time": avg_activation_to_hit_time,
        "all_activation_times": all_activation_times,
        "all_activation_distances": all_activation_distances,
        "all_notch_times": all_notch_times,
        "all_angle_change_times": all_angle_change_times,

        "final_mx": final_mx,
        "final_my": final_my,
        "final_mz": final_mz,
        "final_tx": final_tx,
        "final_ty": final_ty,
        "final_tz": final_tz,

        "final_time": final_time,
        "final_missile_mach": final_missile_mach,
        "final_target_mach": final_target_mach,
        "final_missile_ms": final_missile_ms,
        "final_target_ms": final_target_ms,
        "final_distance": final_distance,
        "final_phase": final_phase,
        "final_target_phase": final_target_phase,
        "final_drag_factor": final_drag_factor,
        "final_actual_drag": final_actual_drag,
        "final_turn_rate": final_turn_rate,
        "final_thrust": final_thrust,
        "final_motor_accel": final_motor_accel,
        "final_target_accel": final_target_accel,
        "final_target_accel_perp": final_target_accel_perp,

        "final_ping_point": final_ping_point,
        "final_ping_distance": final_ping_distance,
        "final_ping_time": final_ping_time,

        "avg_ping_point": avg_ping_point,
        "avg_ping_distance": avg_ping_distance,
        "avg_ping_time": avg_ping_time,

        "final_activation_point": final_activation_point,
        "final_activation_time": final_activation_time,
        "final_activation_to_intercept_time": final_activation_to_intercept_time,
        "final_notch_point": final_notch_point,
        "final_angle_change_point": final_angle_change_point,
        "final_notch_side": final_notch_side,

        "final_intercepted": final_intercepted,
        "final_end_time": final_end_time,
        "final_end_distance": final_end_distance,
    }


if run_button:
    result = run_simulation()

    st.subheader("Results")

    if result["all_hit_times"]:
        st.write(f"Average intercept time: **{sum(result['all_hit_times']) / len(result['all_hit_times']):.2f} sec**")
    else:
        st.write("Missile failed to intercept within max simulation time.")

    if result["avg_activation_to_hit_time"] is not None:
        st.write(f"Average seeker activation → intercept time: **{result['avg_activation_to_hit_time']:.2f} sec**")

    if result["all_activation_times"]:
        st.write(f"Missile reached seeker activation range: **{sum(result['all_activation_distances']) / len(result['all_activation_distances']):.2f} km**")
        st.write(f"Time when it reached seeker activation range: **{sum(result['all_activation_times']) / len(result['all_activation_times']):.2f} sec**")

    if notch_mode != 0:
        if result["all_notch_times"]:
            st.write(f"Target started notching at: **{sum(result['all_notch_times']) / len(result['all_notch_times']):.2f} sec**")
        else:
            st.write("Target never started notching.")

    if change_target_angle:
        if result["all_angle_change_times"]:
            st.write(f"Target changed direction at: **{sum(result['all_angle_change_times']) / len(result['all_angle_change_times']):.2f} sec**")
        else:
            st.write("Target never reached the selected altitude for direction change.")

    if has_lpi:
        if runs > 1 and result["avg_ping_distance"] is not None:
            st.write(f"Average first ping distance: **{result['avg_ping_distance']:.2f} km**")
            st.write(f"Average first ping time: **{result['avg_ping_time']:.2f} sec**")
        elif runs == 1 and result["final_ping_distance"] is not None:
            st.write(f"First ping distance: **{result['final_ping_distance']:.2f} km**")
            st.write(f"First ping time: **{result['final_ping_time']:.2f} sec**")
        else:
            st.write("No RWR ping before impact/failure.")

    final_mx = result["final_mx"]
    final_my = result["final_my"]
    final_mz = result["final_mz"]
    final_tx = result["final_tx"]
    final_ty = result["final_ty"]
    final_tz = result["final_tz"]

    if final_mx and final_tx:
        fig = go.Figure()

        missile_hover = []

        for i in range(len(final_mx)):
            manual_launch_text = "No"
            if use_manual_launch:
                manual_launch_text = (
                    f"Yes<br>"
                    f"Launch horizontal offset: {launch_horizontal_offset_deg:.1f}°<br>"
                    f"Launch vertical angle: {launch_vertical_angle_deg:.1f}°"
                )

            missile_hover.append(
                f"Missile<br>"
                f"Time: {result['final_time'][i]:.2f}s<br>"
                f"Phase: {result['final_phase'][i]}<br>"
                f"Speed: Mach {result['final_missile_mach'][i]:.2f}<br>"
                f"Speed: {result['final_missile_ms'][i]:.0f} m/s<br>"
                f"Manual launch: {manual_launch_text}<br>"
                f"Launch platform Mach: {launch_platform_mach:.2f}<br>"
                f"Motor thrust: {result['final_thrust'][i]:.0f} N<br>"
                f"Motor accel: {result['final_motor_accel'][i]:.1f} m/s²<br>"
                f"Target accel used by APN: {result['final_target_accel'][i]:.1f} m/s²<br>"
                f"Target accel perpendicular to LOS: {result['final_target_accel_perp'][i]:.1f} m/s²<br>"
                f"Missile mass: {missile_mass_kg:.1f} kg<br>"
                f"Max G: {missile_max_g:.1f}G<br>"
                f"Turn rate: {result['final_turn_rate'][i]:.1f}°/s<br>"
                f"Gravity: {'ON' if USE_GRAVITY else 'OFF'}<br>"
                f"Base drag: {missile_drag_mach:.4f} Mach/sec<br>"
                f"Altitude drag factor: {result['final_drag_factor'][i]:.2f}<br>"
                f"Actual drag: {result['final_actual_drag'][i]:.4f} Mach/sec<br>"
                f"True 3D distance to target: {result['final_distance'][i]:.2f} km<br>"
                f"X: {final_mx[i]:.2f} km<br>"
                f"Y: {final_my[i]:.2f} km<br>"
                f"Alt: {final_mz[i]:.2f} km"
            )

        target_hover = []

        for i in range(len(final_tx)):
            target_hover.append(
                f"Target<br>"
                f"Time: {result['final_time'][i]:.2f}s<br>"
                f"Phase: {result['final_target_phase'][i]}<br>"
                f"Speed: Mach {result['final_target_mach'][i]:.2f}<br>"
                f"Speed: {result['final_target_ms'][i]:.0f} m/s<br>"
                f"Target heading input: {target_heading:.1f}° relative<br>"
                f"Target accel: {result['final_target_accel'][i]:.1f} m/s²<br>"
                f"Target accel perpendicular to LOS: {result['final_target_accel_perp'][i]:.1f} m/s²<br>"
                f"Max speed: Mach {target_max_mach:.2f}<br>"
                f"Turn rate setting: {target_turn_rate_deg:.1f}°/s<br>"
                f"True 3D distance to missile: {result['final_distance'][i]:.2f} km<br>"
                f"X: {final_tx[i]:.2f} km<br>"
                f"Y: {final_ty[i]:.2f} km<br>"
                f"Alt: {final_tz[i]:.2f} km"
            )

        terminal_name = f"Document-style APN, N={nav_constant}"
        guidance_name = f"Auto loft aim {loft_angle}° + {terminal_name}" if use_loft else terminal_name

        fig.add_trace(go.Scatter3d(
            x=final_mx,
            y=final_my,
            z=final_mz,
            mode="lines",
            name="Missile path",
            line=dict(width=6),
            hovertext=missile_hover,
            hoverinfo="text"
        ))

        fig.add_trace(go.Scatter3d(
            x=final_tx,
            y=final_ty,
            z=final_tz,
            mode="lines",
            name="Target path",
            line=dict(width=6),
            hovertext=target_hover,
            hoverinfo="text"
        ))

        launch_hover = [
            f"Missile launch<br>"
            f"Launch platform Mach: {launch_platform_mach:.2f}<br>"
            f"Starting horizontal range: {start_horizontal_range:.2f} km<br>"
            f"Alt: {missile_altitude:.2f} km"
        ]

        if use_manual_launch:
            launch_hover = [
                f"Missile launch<br>"
                f"Manual launch direction: Yes<br>"
                f"Horizontal offset: {launch_horizontal_offset_deg:.1f}°<br>"
                f"Vertical angle: {launch_vertical_angle_deg:.1f}°<br>"
                f"Launch platform Mach: {launch_platform_mach:.2f}<br>"
                f"Starting horizontal range: {start_horizontal_range:.2f} km<br>"
                f"Alt: {missile_altitude:.2f} km"
            ]

        fig.add_trace(go.Scatter3d(
            x=[final_mx[0]],
            y=[final_my[0]],
            z=[final_mz[0]],
            mode="markers+text",
            name="Missile launch",
            text=["Missile launch"],
            marker=dict(size=6),
            hovertext=launch_hover,
            hoverinfo="text"
        ))

        fig.add_trace(go.Scatter3d(
            x=[final_tx[0]],
            y=[final_ty[0]],
            z=[final_tz[0]],
            mode="markers+text",
            name="Target start",
            text=["Target start"],
            marker=dict(size=6),
            hovertext=[
                f"Target start<br>"
                f"Target heading: {target_heading:.1f}° relative<br>"
                f"0° = away, 180° = toward, 90° = right, -90° = left<br>"
                f"Mach {target_mach_start:.2f}<br>"
                f"Max Mach {target_max_mach:.2f}<br>"
                f"Turn rate: {target_turn_rate_deg:.1f}°/s<br>"
                f"Alt {target_altitude:.2f} km"
            ],
            hoverinfo="text"
        ))

        if result["final_activation_point"] is not None:
            p = result["final_activation_point"]
            fig.add_trace(go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                name="Seeker activation range marker",
                text=["Seeker range"],
                marker=dict(size=9),
                hovertext=[
                    f"Seeker activation range marker<br>"
                    f"Activation range: {activation_range:.2f} km<br>"
                    f"Time: {result['final_activation_time']:.2f}s"
                ],
                hoverinfo="text"
            ))

        if result["final_notch_point"] is not None:
            p = result["final_notch_point"]
            fig.add_trace(go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                name="Target notch start",
                text=["Notch"],
                marker=dict(size=9),
                hovertext=[
                    f"Target started notching<br>"
                    f"Notch mode: {notch_mode}<br>"
                    f"Side: fixed {result['final_notch_side']}<br>"
                    f"Vertical angle: {notch_vertical_angle:.2f}°<br>"
                    f"Target turn rate: {target_turn_rate_deg:.1f}°/s"
                ],
                hoverinfo="text"
            ))

        if result["final_angle_change_point"] is not None:
            p = result["final_angle_change_point"]
            fig.add_trace(go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                name="Target direction change",
                text=["Target turn"],
                marker=dict(size=9),
                hovertext=[
                    f"Target direction change<br>"
                    f"Changed at altitude: {change_altitude:.2f} km<br>"
                    f"New heading relative: {new_target_heading:.2f}°<br>"
                    f"New vertical angle: {new_target_climb:.2f}°<br>"
                    f"Turn rate: {target_turn_rate_deg:.1f}°/s"
                ],
                hoverinfo="text"
            ))

        if result["final_intercepted"]:
            end_name = "Intercept"
            end_text = "Intercept"

            activation_to_intercept_text = "unknown"
            if result["final_activation_to_intercept_time"] is not None:
                activation_to_intercept_text = f"{result['final_activation_to_intercept_time']:.2f}s"

            end_hover = [
                f"Intercept<br>"
                f"Time: {result['final_end_time']:.2f}s<br>"
                f"Miss distance: {result['final_end_distance'] * 1000:.2f} m<br>"
                f"Seeker activation to intercept: {activation_to_intercept_text}"
            ]
        else:
            end_name = "Simulation end"
            end_text = "End"

            final_dist_text = f"{result['final_end_distance']:.2f} km" if result["final_end_distance"] is not None else "unknown"

            end_hover = [
                f"Simulation ended without intercept<br>"
                f"Final distance: {final_dist_text}<br>"
                f"Time: {result['final_end_time']:.2f}s<br>"
                f"Target X: {final_tx[-1]:.2f} km<br>"
                f"Target Y: {final_ty[-1]:.2f} km<br>"
                f"Target Alt: {final_tz[-1]:.2f} km<br>"
                f"Missile X: {final_mx[-1]:.2f} km<br>"
                f"Missile Y: {final_my[-1]:.2f} km<br>"
                f"Missile Alt: {final_mz[-1]:.2f} km"
            ]

        fig.add_trace(go.Scatter3d(
            x=[final_tx[-1]],
            y=[final_ty[-1]],
            z=[final_tz[-1]],
            mode="markers+text",
            name=end_name,
            text=[end_text],
            marker=dict(size=9),
            hovertext=end_hover,
            hoverinfo="text"
        ))

        if has_lpi and runs == 1 and result["final_ping_point"] is not None:
            p = result["final_ping_point"]
            fig.add_trace(go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                name="First ping",
                text=["First ping"],
                marker=dict(size=8),
                hovertext=[
                    f"First ping<br>"
                    f"Distance: {result['final_ping_distance']:.2f} km<br>"
                    f"Time: {result['final_ping_time']:.2f}s"
                ],
                hoverinfo="text"
            ))

        if has_lpi and runs > 1 and result["avg_ping_point"] is not None:
            p = result["avg_ping_point"]
            fig.add_trace(go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                name="Average first ping",
                text=["Avg ping"],
                marker=dict(size=10),
                hovertext=[
                    f"Average first ping<br>"
                    f"Distance: {result['avg_ping_distance']:.2f} km<br>"
                    f"Time: {result['avg_ping_time']:.2f}s"
                ],
                hoverinfo="text"
            ))

        fig.update_layout(
            title=f"3D Missile Intercept - {guidance_name}",
            scene=dict(
                xaxis_title="X km",
                yaxis_title="Y km",
                zaxis_title="Altitude km",
                aspectmode="data"
            ),
            height=750
        )

        st.plotly_chart(fig, use_container_width=True)

        if show_animation:
            st.subheader("Animated Playback")

            anim_fig = make_animation_figure(
                result,
                animation_frame_skip,
                animation_speed_ms
            )

            if anim_fig is not None:
                st.plotly_chart(anim_fig, use_container_width=True)
            else:
                st.write("Animation could not be created because there was no path data.")

else:
    st.info("Set the values in the sidebar, then click Run simulation.")
