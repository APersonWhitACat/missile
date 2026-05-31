import math
import random
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# Constants
# ----------------------------

MACH_TABLE = {
    1: 336, 2: 332, 3: 328, 4: 324, 5: 320, 6: 316,
    7: 312, 8: 308, 9: 303, 10: 299, 11: 295,
    12: 295, 13: 295, 14: 295, 15: 295, 16: 295, 17: 295,
}

DT = 0.02
MAX_TIME = 300.0
PROXY_FUSE_M = 25.0
GRAVITY = 9.81
USE_GRAVITY = True
NOTCH_BREAK_TIME = 1.2
SEA_LEVEL_DENSITY = 1.225


# ----------------------------
# Vector math
# ----------------------------

def v_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def v_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def v_mul(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def v_mag(a):
    return math.sqrt(v_dot(a, a))


def v_norm(a):
    m = v_mag(a)
    if m < 1e-9:
        return [0.0, 0.0, 0.0]
    return [a[0] / m, a[1] / m, a[2] / m]


def horizontal_norm(v):
    return v_norm([v[0], v[1], 0.0])


# ----------------------------
# Direction helpers
# ----------------------------

def relative_direction_from_line(base_forward, horizontal_relative_deg, vertical_deg):
    forward = horizontal_norm(base_forward)
    if v_mag(forward) < 1e-9:
        forward = [0.0, 1.0, 0.0]

    right = [forward[1], -forward[0], 0.0]
    h = math.radians(horizontal_relative_deg)
    v = math.radians(vertical_deg)

    horizontal_dir = v_add(v_mul(forward, math.cos(h)), v_mul(right, math.sin(h)))
    horizontal_dir = v_norm(horizontal_dir)

    return v_norm([
        horizontal_dir[0] * math.cos(v),
        horizontal_dir[1] * math.cos(v),
        math.sin(v),
    ])


def target_relative_direction(target_pos, missile_pos, heading_relative_deg, vertical_deg):
    away_from_missile = v_sub(target_pos, missile_pos)
    return relative_direction_from_line(away_from_missile, heading_relative_deg, vertical_deg)


def manual_launch_direction(missile_pos, target_pos, horizontal_offset_deg, vertical_angle_deg):
    toward_target = v_sub(target_pos, missile_pos)
    return relative_direction_from_line(toward_target, horizontal_offset_deg, vertical_angle_deg)


# ----------------------------
# Drag model
# ----------------------------

def air_density_factor(alt_km):
    # Dev/game-style density curve. altitude is meters in the source formula.
    altitude_m = max(0.0, alt_km * 1000.0)
    if altitude_m > 40000.0:
        return 0.0
    return max(0.0, (1.0 - 2.25577e-5 * altitude_m) ** 5.25588)


def air_density_kg_m3(alt_km):
    return SEA_LEVEL_DENSITY * air_density_factor(alt_km)


def frontal_area_m2(diameter_m):
    radius = max(0.001, diameter_m / 2.0)
    return math.pi * radius * radius


def drag_force_newtons(speed_ms, alt_km, diameter_m, cd):
    rho = air_density_kg_m3(alt_km)
    area = frontal_area_m2(diameter_m)
    return 0.5 * rho * speed_ms * speed_ms * cd * area


def turn_drag_force_newtons(base_drag_force, turn_rate_deg_s, turn_drag_multiplier):
    if turn_drag_multiplier <= 0.0:
        return 0.0
    turn_factor = (max(0.0, turn_rate_deg_s) / 100.0) ** 2
    return base_drag_force * turn_drag_multiplier * turn_factor


# ----------------------------
# Sim helpers
# ----------------------------

def closest_distance_between_steps(prev_target, prev_missile, new_target, new_missile):
    r0 = v_sub(prev_target, prev_missile)
    r1 = v_sub(new_target, new_missile)
    dr = v_sub(r1, r0)
    dr_mag2 = v_dot(dr, dr)
    if dr_mag2 < 1e-12:
        return v_mag(r1)
    t = -v_dot(r0, dr) / dr_mag2
    t = max(0.0, min(1.0, t))
    return v_mag(v_add(r0, v_mul(dr, t)))


def turn_toward_direction(old_dir, desired_dir, turn_rate_deg_per_sec, dt):
    old_dir = v_norm(old_dir)
    desired_dir = v_norm(desired_dir)
    dot = max(-1.0, min(1.0, v_dot(old_dir, desired_dir)))
    angle = math.acos(dot)
    if angle < 1e-6:
        return desired_dir
    max_turn = math.radians(turn_rate_deg_per_sec) * dt
    blend = min(1.0, max_turn / angle)
    return v_norm([
        old_dir[0] * (1.0 - blend) + desired_dir[0] * blend,
        old_dir[1] * (1.0 - blend) + desired_dir[1] * blend,
        old_dir[2] * (1.0 - blend) + desired_dir[2] * blend,
    ])


def limit_missile_g(old_vel, commanded_vel, max_g, dt):
    old_speed = v_mag(old_vel)
    commanded_speed = v_mag(commanded_vel)
    if old_speed < 1e-6 or commanded_speed < 1e-6:
        return commanded_vel

    old_dir = v_norm(old_vel)
    commanded_dir = v_norm(commanded_vel)
    max_lat_accel = max_g * GRAVITY
    max_turn_rad = (max_lat_accel / max(old_speed, 1.0)) * dt

    dot = max(-1.0, min(1.0, v_dot(old_dir, commanded_dir)))
    angle = math.acos(dot)
    if angle <= max_turn_rad:
        return commanded_vel

    blend = max_turn_rad / max(angle, 1e-9)
    limited_dir = v_norm([
        old_dir[0] * (1.0 - blend) + commanded_dir[0] * blend,
        old_dir[1] * (1.0 - blend) + commanded_dir[1] * blend,
        old_dir[2] * (1.0 - blend) + commanded_dir[2] * blend,
    ])
    return v_mul(limited_dir, commanded_speed)


def estimate_total_fuel_from_isp(first_thrust, first_time, second_thrust, second_time, isp_seconds):
    total_impulse = first_thrust * first_time + second_thrust * second_time
    if total_impulse <= 0.0 or isp_seconds <= 0.0:
        return 0.0
    return total_impulse / (isp_seconds * GRAVITY)


def split_fuel_by_impulse(total_fuel, first_thrust, first_time, second_thrust, second_time):
    first_impulse = first_thrust * first_time
    second_impulse = second_thrust * second_time
    total_impulse = first_impulse + second_impulse
    if total_impulse <= 0.0 or total_fuel <= 0.0:
        return 0.0, 0.0
    return (
        total_fuel * first_impulse / total_impulse,
        total_fuel * second_impulse / total_impulse,
    )


def choose_notch_side(target_pos, missile_pos, current_target_dir):
    to_missile = [missile_pos[0] - target_pos[0], missile_pos[1] - target_pos[1], 0.0]
    los = v_norm(to_missile)
    if v_mag(los) < 1e-9:
        return "right"

    right_notch = [los[1], -los[0], 0.0]
    left_notch = [-los[1], los[0], 0.0]

    current_horizontal = v_norm([current_target_dir[0], current_target_dir[1], 0.0])
    if v_mag(current_horizontal) < 1e-9:
        current_horizontal = [1.0, 0.0, 0.0]

    if v_dot(left_notch, current_horizontal) > v_dot(right_notch, current_horizontal):
        return "left"
    return "right"


def lpi_detection_chance(distance_m, base_lpi):
    if distance_m <= 3000.0:
        return 1.0
    if distance_m >= 8000.0:
        return base_lpi
    closeness = (8000.0 - distance_m) / 5000.0
    closeness = max(0.0, min(1.0, closeness))
    return base_lpi + closeness * (1.0 - base_lpi)


def estimate_tti_seconds(target_pos, missile_pos, target_vel, missile_vel):
    rel_pos = v_sub(target_pos, missile_pos)
    rel_vel = v_sub(target_vel, missile_vel)
    distance = v_mag(rel_pos)
    if distance < 1e-6:
        return 0.0
    los = v_norm(rel_pos)
    closing_speed = -v_dot(rel_vel, los)
    if closing_speed <= 1.0:
        return None
    return distance / closing_speed


# ----------------------------
# Animation
# ----------------------------

def make_animation_figure(result, frame_skip, play_speed_ms):
    mx = result["final_mx"]
    my = result["final_my"]
    mz = result["final_mz"]
    tx = result["final_tx"]
    ty = result["final_ty"]
    tz = result["final_tz"]
    t = result["final_time"]

    if not mx or not tx:
        return None

    frame_skip = max(1, int(frame_skip))
    play_speed_ms = max(5, int(play_speed_ms))

    indices = list(range(0, len(mx), frame_skip))
    if indices[-1] != len(mx) - 1:
        indices.append(len(mx) - 1)

    all_x = mx + tx
    all_y = my + ty
    all_z = mz + tz
    pad_x = max(1.0, (max(all_x) - min(all_x)) * 0.08)
    pad_y = max(1.0, (max(all_y) - min(all_y)) * 0.08)
    pad_z = max(1.0, (max(all_z) - min(all_z)) * 0.08)

    fig = go.Figure(
        data=[
            go.Scatter3d(x=[mx[0]], y=[my[0]], z=[mz[0]], mode="lines", name="Missile trail", line=dict(width=6)),
            go.Scatter3d(x=[tx[0]], y=[ty[0]], z=[tz[0]], mode="lines", name="Target trail", line=dict(width=6)),
            go.Scatter3d(x=[mx[0]], y=[my[0]], z=[mz[0]], mode="markers+text", name="Missile", text=["Missile"], marker=dict(size=6)),
            go.Scatter3d(x=[tx[0]], y=[ty[0]], z=[tz[0]], mode="markers+text", name="Target", text=["Target"], marker=dict(size=6)),
        ]
    )

    frames = []
    for i in indices:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=mx[:i + 1], y=my[:i + 1], z=mz[:i + 1], mode="lines", name="Missile trail", line=dict(width=6)),
                    go.Scatter3d(x=tx[:i + 1], y=ty[:i + 1], z=tz[:i + 1], mode="lines", name="Target trail", line=dict(width=6)),
                    go.Scatter3d(x=[mx[i]], y=[my[i]], z=[mz[i]], mode="markers+text", name="Missile", text=["Missile"], marker=dict(size=6), hovertext=[f"Missile<br>Time: {t[i]:.2f}s"], hoverinfo="text"),
                    go.Scatter3d(x=[tx[i]], y=[ty[i]], z=[tz[i]], mode="markers+text", name="Target", text=["Target"], marker=dict(size=6), hovertext=[f"Target<br>Time: {t[i]:.2f}s"], hoverinfo="text"),
                ],
                name=str(i),
            )
        )
    fig.frames = frames

    slider_steps = []
    for i in indices:
        slider_steps.append({
            "args": [[str(i)], {"frame": {"duration": play_speed_ms, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": f"{t[i]:.1f}s",
            "method": "animate",
        })

    fig.update_layout(
        title="Animated Playback",
        scene=dict(
            xaxis_title="X km",
            yaxis_title="Y km",
            zaxis_title="Altitude km",
            xaxis=dict(range=[min(all_x) - pad_x, max(all_x) + pad_x]),
            yaxis=dict(range=[min(all_y) - pad_y, max(all_y) + pad_y]),
            zaxis=dict(range=[min(all_z) - pad_z, max(all_z) + pad_z]),
            aspectmode="data",
        ),
        height=750,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05,
            "y": 0,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": play_speed_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{"active": 0, "currentvalue": {"prefix": "Sim time: "}, "pad": {"t": 50}, "steps": slider_steps}],
    )
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="3D Missile Intercept Simulator", layout="wide")
st.title("3D Missile Intercept Simulator")

with st.sidebar:
    st.header("Target Properties")
    target_altitude = st.number_input("Target altitude km", value=12.0, step=0.5)
    target_mach_start = st.number_input("Target speed Mach", value=1.0, step=0.1)
    target_accel_mach = st.number_input("Target acceleration Mach/sec", value=0.0, step=0.01)
    target_max_mach = st.number_input("Target maximum speed Mach", value=float(target_mach_start), step=0.1)

    target_heading = st.number_input("Target heading relative to missile degrees, 0=away, 180=toward, 90=right, -90=left", value=0.0, step=5.0)
    target_climb = st.number_input("Target vertical angle degrees, +up / -down", value=0.0, step=1.0)
    target_turn_rate_deg = st.number_input("Target turn rate deg/sec", value=45.0, min_value=1.0, step=1.0)

    if target_max_mach < target_mach_start:
        target_max_mach = target_mach_start
        st.warning("Target max Mach was raised to match starting Mach.")

    change_target_angle = st.checkbox("Change target direction after reaching altitude", value=False)
    change_altitude = None
    new_target_heading = None
    new_target_climb = None

    if change_target_angle:
        change_altitude = st.number_input("Altitude where target changes direction km", value=float(target_altitude), step=0.5)
        new_target_heading = st.number_input("New target heading relative to missile degrees, 0=away, 180=toward, 90=right, -90=left", value=float(target_heading), step=5.0)
        new_target_climb = st.number_input("New target vertical angle degrees, +up / -down", value=float(target_climb), step=1.0)

    st.header("Target Notching")
    notch_mode = st.selectbox(
        "Target notch mode",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 = No notch",
            1: "1 = Notch at chosen missile distance",
            2: "2 = Notch after first RWR ping, LPI only",
            3: "3 = Notch at seeker activation range",
        }[x],
    )

    notch_distance = None
    notch_vertical_angle = 0.0
    if notch_mode != 0:
        if notch_mode == 1:
            notch_distance = st.number_input("Start notching when missile is this far away km", value=8.0, step=0.5)
        notch_vertical_angle = st.number_input("Target vertical angle while notching degrees, +up / -down", value=0.0, step=1.0)
        notch_vertical_angle = max(-60.0, min(60.0, notch_vertical_angle))

    st.header("Missile / Launch Platform")
    missile_altitude = st.number_input("Launch platform altitude km", value=float(target_altitude), step=0.5)
    launch_platform_mach = st.number_input("Launch platform speed Mach", value=1.2, step=0.1)

    use_manual_launch = st.checkbox("Use manual launch direction", value=False)
    launch_horizontal_offset_deg = 0.0
    launch_vertical_angle_deg = 0.0
    if use_manual_launch:
        launch_horizontal_offset_deg = st.number_input("Launch horizontal offset degrees, 0=toward target, 90=right, -90=left", value=0.0, step=5.0)
        launch_vertical_angle_deg = st.number_input("Launch vertical angle degrees, +up / -down", value=20.0, step=5.0)

    missile_mass_kg = st.number_input("Missile total mass kg", value=168.0, min_value=1.0, step=1.0)

    st.header("Motor / Dual Pulse")
    first_pulse_thrust_n = st.number_input("First pulse thrust N", value=19500.0, min_value=0.0, step=500.0)
    first_pulse_burn_time = st.number_input("First pulse burn time s", value=8.0, min_value=0.0, step=0.5)

    use_dual_pulse = st.checkbox("Use dual pulse second motor", value=False)
    second_pulse_thrust_n = 0.0
    second_pulse_burn_time = 0.0
    second_pulse_trigger_mode = "After flight time"
    second_pulse_trigger_time = 20.0
    second_pulse_trigger_distance = 20.0
    second_pulse_trigger_tti = 40.0

    if use_dual_pulse:
        second_pulse_thrust_n = st.number_input("Second pulse thrust N", value=12000.0, min_value=0.0, step=500.0)
        second_pulse_burn_time = st.number_input("Second pulse burn time s", value=4.0, min_value=0.0, step=0.5)
        second_pulse_trigger_mode = st.selectbox("Second pulse trigger mode", options=["After flight time", "At target distance", "Time to target estimate"])

        if second_pulse_trigger_mode == "After flight time":
            second_pulse_trigger_time = st.number_input("Start second pulse after flight time s", value=20.0, min_value=0.0, step=1.0)
        elif second_pulse_trigger_mode == "At target distance":
            second_pulse_trigger_distance = st.number_input("Start second pulse when target distance is km", value=20.0, min_value=0.0, step=1.0)
        elif second_pulse_trigger_mode == "Time to target estimate":
            second_pulse_trigger_tti = st.number_input("Start second pulse when estimated time to target is s", value=40.0, min_value=0.0, step=1.0)

    st.header("Fuel / Mass Loss")
    fuel_mass_mode = st.selectbox("Fuel mass mode", options=["No mass loss", "Known fuel mass", "Estimate from Isp"])
    known_fuel_mass_kg = 0.0
    isp_seconds = 240.0

    if fuel_mass_mode == "Known fuel mass":
        known_fuel_mass_kg = st.number_input("Known total fuel mass kg", value=0.0, min_value=0.0, max_value=float(missile_mass_kg * 0.95), step=1.0)
    if fuel_mass_mode == "Estimate from Isp":
        isp_seconds = st.number_input("Specific impulse Isp seconds", value=240.0, min_value=1.0, step=5.0)

    second_fuel_thrust_for_calc = second_pulse_thrust_n if use_dual_pulse else 0.0
    second_fuel_time_for_calc = second_pulse_burn_time if use_dual_pulse else 0.0

    if fuel_mass_mode == "No mass loss":
        displayed_total_fuel = 0.0
    elif fuel_mass_mode == "Known fuel mass":
        displayed_total_fuel = min(known_fuel_mass_kg, missile_mass_kg * 0.95)
    else:
        displayed_total_fuel = estimate_total_fuel_from_isp(first_pulse_thrust_n, first_pulse_burn_time, second_fuel_thrust_for_calc, second_fuel_time_for_calc, isp_seconds)
        displayed_total_fuel = min(displayed_total_fuel, missile_mass_kg * 0.80)

    displayed_dry_mass = missile_mass_kg - displayed_total_fuel
    displayed_first_fuel, displayed_second_fuel = split_fuel_by_impulse(displayed_total_fuel, first_pulse_thrust_n, first_pulse_burn_time, second_fuel_thrust_for_calc, second_fuel_time_for_calc)

    st.caption(f"Fuel mass: {displayed_total_fuel:.1f} kg")
    st.caption(f"Dry mass: {displayed_dry_mass:.1f} kg")
    st.caption(f"First pulse fuel: {displayed_first_fuel:.1f} kg")
    if use_dual_pulse:
        st.caption(f"Second pulse fuel: {displayed_second_fuel:.1f} kg")

    missile_max_g = st.number_input("Missile max G", value=40.0, min_value=1.0, step=1.0)

    st.header("Missile Drag")
    missile_diameter_m = st.number_input("Missile diameter m", value=0.178, min_value=0.01, step=0.001, format="%.3f", help="Body diameter of the missile.")
    drag_coefficient_cd = st.number_input("Drag coefficient Cd", value=0.45, min_value=0.01, step=0.05, format="%.2f", help="Higher = more drag. Use this as the main tuning value.")
    turn_drag_multiplier = st.number_input("Turn drag multiplier", value=1.0, min_value=0.0, step=0.1, format="%.2f", help="Extra drag while turning. 0 disables turn drag.")
    st.caption("Drag uses force: air density × speed² × Cd × frontal area. No Mach/sec drag.")

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
        lpi_value = st.number_input("Base LPI value", value=0.07, min_value=0.0, max_value=1.0, step=0.01)
        runs = st.number_input("Simulation runs", value=1, min_value=1, max_value=200, step=1)

    st.header("Playback")
    show_animation = st.checkbox("Show animated playback", value=False)
    animation_frame_skip = 20
    animation_speed_ms = 60
    if show_animation:
        animation_speed_ms = st.number_input("Playback speed ms per frame", value=60, min_value=5, max_value=1000, step=5)
        animation_frame_skip = st.number_input("Animation frame skip", value=20, min_value=1, max_value=200, step=1)

    run_button = st.button("Run simulation", type="primary")


# ----------------------------
# Simulation
# ----------------------------

def run_simulation():
    alt_key = max(1, min(17, round(target_altitude)))
    sound_speed = MACH_TABLE[alt_key]
    target_max_speed = target_max_mach * sound_speed

    second_calc_thrust = second_pulse_thrust_n if use_dual_pulse else 0.0
    second_calc_time = second_pulse_burn_time if use_dual_pulse else 0.0

    if fuel_mass_mode == "No mass loss":
        total_fuel_kg = 0.0
    elif fuel_mass_mode == "Known fuel mass":
        total_fuel_kg = min(known_fuel_mass_kg, missile_mass_kg * 0.95)
    else:
        total_fuel_kg = estimate_total_fuel_from_isp(first_pulse_thrust_n, first_pulse_burn_time, second_calc_thrust, second_calc_time, isp_seconds)
        total_fuel_kg = min(total_fuel_kg, missile_mass_kg * 0.80)

    dry_mass_kg = missile_mass_kg - total_fuel_kg
    initial_first_fuel_kg, initial_second_fuel_kg = split_fuel_by_impulse(total_fuel_kg, first_pulse_thrust_n, first_pulse_burn_time, second_calc_thrust, second_calc_time)

    all_hit_times = []
    all_activation_to_hit_times = []
    all_first_ping_distances = []
    all_first_ping_times = []
    all_first_ping_points = []
    all_activation_times = []
    all_activation_distances = []
    all_notch_times = []
    all_angle_change_times = []
    all_second_pulse_times = []
    all_second_pulse_tti = []

    final_data = None

    for run in range(int(runs)):
        remaining_first_fuel = initial_first_fuel_kg
        remaining_second_fuel = initial_second_fuel_kg
        second_pulse_started = False
        second_pulse_start_time = None

        missile_mach = launch_platform_mach
        target_mach = target_mach_start
        missile_speed = missile_mach * sound_speed
        target_speed = target_mach * sound_speed

        target_pos = [0.0, 0.0, target_altitude * 1000.0]
        missile_pos = [0.0, -start_horizontal_range * 1000.0, missile_altitude * 1000.0]

        current_target_dir = target_relative_direction(target_pos, missile_pos, target_heading, target_climb)
        target_vel = v_mul(current_target_dir, target_speed)

        if use_manual_launch:
            missile_dir = manual_launch_direction(missile_pos, target_pos, launch_horizontal_offset_deg, launch_vertical_angle_deg)
        else:
            missile_dir = v_norm(v_sub(target_pos, missile_pos))
        missile_vel = v_mul(missile_dir, missile_speed)

        time = 0.0
        activation_time = None
        ping_timer = 0.0
        rwr_ping_visible_timer = 0.0
        reached_activation_range = False
        angle_changed = False
        notch_started = False
        notch_start_time = None
        chosen_notch_side = None
        intercepted = False
        first_ping_distance = None
        first_ping_time = None
        first_ping_point = None

        data = {
            "mx": [], "my": [], "mz": [], "tx": [], "ty": [], "tz": [], "time": [],
            "missile_mach": [], "target_mach": [], "missile_ms": [], "target_ms": [],
            "distance": [], "phase": [], "target_phase": [], "air_density_factor": [],
            "air_density_kg_m3": [], "aero_drag_force": [], "turn_drag_force": [],
            "total_drag_force": [], "drag_accel": [], "turn_rate": [], "thrust": [],
            "motor_accel": [], "target_accel": [], "target_accel_perp": [],
            "current_mass": [], "remaining_fuel": [], "tti": [],
        }

        markers = {
            "activation_point": None, "activation_time": None, "notch_point": None,
            "angle_change_point": None, "notch_side": None, "second_pulse_point": None,
            "second_pulse_time": None, "second_pulse_tti": None, "intercepted": False,
            "end_time": None, "end_distance": None, "activation_to_intercept_time": None,
            "ping_point": None, "ping_distance": None, "ping_time": None,
        }

        def record_point(phase_text, target_phase_text, current_stage, current_thrust_n,
                         motor_accel_ms2, actual_target_accel_vec, target_accel_perp_mag,
                         current_mass_kg, remaining_total_fuel, actual_turn_rate,
                         base_drag_force, extra_turn_drag_force, total_drag_force,
                         drag_accel_ms2, current_tti):
            data["mx"].append(missile_pos[0] / 1000.0)
            data["my"].append(missile_pos[1] / 1000.0)
            data["mz"].append(missile_pos[2] / 1000.0)
            data["tx"].append(target_pos[0] / 1000.0)
            data["ty"].append(target_pos[1] / 1000.0)
            data["tz"].append(target_pos[2] / 1000.0)
            data["time"].append(time)
            data["missile_mach"].append(missile_mach)
            data["target_mach"].append(target_mach)
            data["missile_ms"].append(missile_speed)
            data["target_ms"].append(target_speed)
            data["distance"].append(v_mag(v_sub(target_pos, missile_pos)) / 1000.0)
            data["phase"].append(f"{phase_text}, {current_stage}")
            data["target_phase"].append(target_phase_text)
            alt_km = missile_pos[2] / 1000.0
            data["air_density_factor"].append(air_density_factor(alt_km))
            data["air_density_kg_m3"].append(air_density_kg_m3(alt_km))
            data["aero_drag_force"].append(base_drag_force)
            data["turn_drag_force"].append(extra_turn_drag_force)
            data["total_drag_force"].append(total_drag_force)
            data["drag_accel"].append(drag_accel_ms2)
            data["turn_rate"].append(actual_turn_rate)
            data["thrust"].append(current_thrust_n)
            data["motor_accel"].append(motor_accel_ms2)
            data["target_accel"].append(v_mag(actual_target_accel_vec))
            data["target_accel_perp"].append(target_accel_perp_mag)
            data["current_mass"].append(current_mass_kg)
            data["remaining_fuel"].append(remaining_total_fuel)
            data["tti"].append(current_tti)

        while time <= MAX_TIME:
            prev_target_pos = target_pos[:]
            prev_missile_pos = missile_pos[:]
            prev_missile_vel = missile_vel[:]
            prev_target_vel = target_vel[:]

            rel_pos = v_sub(target_pos, missile_pos)
            distance = v_mag(rel_pos)

            if not reached_activation_range and distance <= activation_range * 1000.0:
                reached_activation_range = True
                activation_time = time
                activation_point = (missile_pos[0] / 1000.0, missile_pos[1] / 1000.0, missile_pos[2] / 1000.0)
                all_activation_times.append(time)
                all_activation_distances.append(distance / 1000.0)
                markers["activation_point"] = activation_point
                markers["activation_time"] = time

            if change_target_angle and not angle_changed and not notch_started:
                current_alt_km = target_pos[2] / 1000.0
                should_change = False
                if target_climb >= 0 and current_alt_km >= change_altitude:
                    should_change = True
                elif target_climb < 0 and current_alt_km <= change_altitude:
                    should_change = True

                if should_change:
                    desired_target_dir = target_relative_direction(target_pos, missile_pos, new_target_heading, new_target_climb)
                    current_target_dir = turn_toward_direction(current_target_dir, desired_target_dir, target_turn_rate_deg, DT)
                    target_vel = v_mul(current_target_dir, target_speed)
                    if v_dot(current_target_dir, desired_target_dir) > 0.999:
                        angle_changed = True
                    if markers["angle_change_point"] is None:
                        all_angle_change_times.append(time)
                        markers["angle_change_point"] = (target_pos[0] / 1000.0, target_pos[1] / 1000.0, target_pos[2] / 1000.0)

            if notch_mode != 0 and not notch_started:
                notch_triggered = False
                if notch_mode == 1 and distance <= notch_distance * 1000.0:
                    notch_triggered = True
                elif notch_mode == 2 and has_lpi and first_ping_distance is not None:
                    notch_triggered = True
                elif notch_mode == 3 and reached_activation_range:
                    notch_triggered = True

                if notch_triggered:
                    notch_started = True
                    notch_start_time = time
                    chosen_notch_side = choose_notch_side(target_pos, missile_pos, current_target_dir)
                    all_notch_times.append(time)
                    markers["notch_point"] = (target_pos[0] / 1000.0, target_pos[1] / 1000.0, target_pos[2] / 1000.0)
                    markers["notch_side"] = chosen_notch_side

            if notch_started:
                should_update_notch = False
                if notch_mode in [1, 3]:
                    should_update_notch = True
                elif notch_mode == 2 and has_lpi and rwr_ping_visible_timer > 0:
                    should_update_notch = True

                if should_update_notch:
                    to_missile = [missile_pos[0] - target_pos[0], missile_pos[1] - target_pos[1], 0.0]
                    los = v_norm(to_missile)
                    if v_mag(los) < 1e-9:
                        los = [0.0, -1.0, 0.0]
                    desired_horizontal = [-los[1], los[0], 0.0] if chosen_notch_side == "left" else [los[1], -los[0], 0.0]
                    v_ang = math.radians(notch_vertical_angle)
                    desired_notch_dir = v_norm([desired_horizontal[0] * math.cos(v_ang), desired_horizontal[1] * math.cos(v_ang), math.sin(v_ang)])
                    notch_elapsed = time - notch_start_time
                    notch_turn_rate = target_turn_rate_deg * 2.0 if notch_elapsed < NOTCH_BREAK_TIME else target_turn_rate_deg
                    current_target_dir = turn_toward_direction(current_target_dir, desired_notch_dir, notch_turn_rate, DT)

            target_accel_ms = target_accel_mach * sound_speed
            target_speed = max(0.0, min(target_speed + target_accel_ms * DT, target_max_speed))
            target_mach = target_speed / sound_speed
            target_vel = v_mul(current_target_dir, target_speed)
            target_pos = v_add(target_pos, v_mul(target_vel, DT))

            actual_target_accel_vec = v_mul(v_sub(target_vel, prev_target_vel), 1.0 / max(DT, 1e-9))

            rel_pos = v_sub(target_pos, missile_pos)
            rel_vel = v_sub(target_vel, missile_vel)
            distance = v_mag(rel_pos)
            if distance < 1e-9:
                intercepted = True
                break

            current_tti = estimate_tti_seconds(target_pos, missile_pos, target_vel, missile_vel)

            if use_loft and not reached_activation_range:
                current_range_km = distance / 1000.0
                if start_horizontal_range > activation_range:
                    loft_fraction = (current_range_km - activation_range) / (start_horizontal_range - activation_range)
                else:
                    loft_fraction = 0.0
                loft_fraction = max(0.0, min(1.0, loft_fraction))

                horizontal_distance = math.sqrt((target_pos[0] - missile_pos[0]) ** 2 + (target_pos[1] - missile_pos[1]) ** 2)
                max_loft_offset = math.tan(math.radians(loft_angle)) * horizontal_distance
                max_loft_offset = min(max_loft_offset, 50000.0)
                current_loft_offset = max_loft_offset * loft_fraction * loft_strength
                loft_aim_point = [target_pos[0], target_pos[1], target_pos[2] + current_loft_offset]
                desired_dir = v_norm(v_sub(loft_aim_point, missile_pos))
                current_loft_angle = math.degrees(math.atan2(loft_aim_point[2] - missile_pos[2], max(horizontal_distance, 1.0)))
                phase = f"Loft {current_loft_angle:.1f}°"
                commanded_missile_vel = v_mul(desired_dir, missile_speed)
                target_accel_perp_mag = 0.0
            else:
                phase = "APN"
                omega = v_mul(v_cross(rel_pos, rel_vel), 1.0 / max(distance * distance, 1e-9))
                closing_speed = -v_dot(rel_pos, rel_vel) / max(distance, 1e-9)
                closing_speed = max(closing_speed, 0.0)
                los_unit = v_norm(rel_pos)
                pn_accel = v_mul(v_cross(los_unit, omega), -nav_constant * closing_speed)
                target_accel_parallel = v_mul(los_unit, v_dot(actual_target_accel_vec, los_unit))
                target_accel_perp = v_sub(actual_target_accel_vec, target_accel_parallel)
                target_accel_perp_mag = v_mag(target_accel_perp)
                apn_accel = v_mul(target_accel_perp, apn_gain * nav_constant / 2.0)
                commanded_accel = v_add(pn_accel, apn_accel)
                commanded_missile_vel = v_add(missile_vel, v_mul(commanded_accel, DT))

            missile_vel = limit_missile_g(missile_vel, commanded_missile_vel, missile_max_g, DT)

            first_pulse_burning = time < first_pulse_burn_time

            if use_dual_pulse and not second_pulse_started and not first_pulse_burning:
                should_start_second_pulse = False
                if second_pulse_trigger_mode == "After flight time" and time >= second_pulse_trigger_time:
                    should_start_second_pulse = True
                elif second_pulse_trigger_mode == "At target distance" and distance <= second_pulse_trigger_distance * 1000.0:
                    should_start_second_pulse = True
                elif second_pulse_trigger_mode == "Time to target estimate" and current_tti is not None and current_tti <= second_pulse_trigger_tti:
                    should_start_second_pulse = True

                if should_start_second_pulse:
                    second_pulse_started = True
                    second_pulse_start_time = time
                    second_pulse_point = (missile_pos[0] / 1000.0, missile_pos[1] / 1000.0, missile_pos[2] / 1000.0)
                    all_second_pulse_times.append(time)
                    if current_tti is not None:
                        all_second_pulse_tti.append(current_tti)
                    markers["second_pulse_point"] = second_pulse_point
                    markers["second_pulse_time"] = time
                    markers["second_pulse_tti"] = current_tti

            second_pulse_burning = False
            if use_dual_pulse and second_pulse_started and second_pulse_start_time is not None:
                second_pulse_elapsed = time - second_pulse_start_time
                if second_pulse_elapsed < second_pulse_burn_time:
                    second_pulse_burning = True

            current_stage = "off"
            current_thrust_n = 0.0
            if first_pulse_burning:
                current_stage = "first pulse"
                current_thrust_n = first_pulse_thrust_n
            elif second_pulse_burning:
                current_stage = "second pulse"
                current_thrust_n = second_pulse_thrust_n

            if fuel_mass_mode != "No mass loss":
                if current_stage == "first pulse":
                    if first_pulse_burn_time > 0 and initial_first_fuel_kg > 0 and remaining_first_fuel > 0:
                        burn_rate = initial_first_fuel_kg / first_pulse_burn_time
                        fuel_to_burn = min(remaining_first_fuel, burn_rate * DT)
                        remaining_first_fuel -= fuel_to_burn
                    else:
                        current_thrust_n = 0.0
                elif current_stage == "second pulse":
                    if second_pulse_burn_time > 0 and initial_second_fuel_kg > 0 and remaining_second_fuel > 0:
                        burn_rate = initial_second_fuel_kg / second_pulse_burn_time
                        fuel_to_burn = min(remaining_second_fuel, burn_rate * DT)
                        remaining_second_fuel -= fuel_to_burn
                    else:
                        current_thrust_n = 0.0

                if current_stage == "first pulse" and remaining_first_fuel <= 0:
                    current_thrust_n = 0.0
                if current_stage == "second pulse" and remaining_second_fuel <= 0:
                    current_thrust_n = 0.0

                current_mass_kg = dry_mass_kg + remaining_first_fuel + remaining_second_fuel
                remaining_total_fuel = remaining_first_fuel + remaining_second_fuel
            else:
                current_mass_kg = missile_mass_kg
                remaining_total_fuel = 0.0

            current_mass_kg = max(current_mass_kg, 1.0)
            motor_accel_ms2 = current_thrust_n / current_mass_kg

            thrust_dir = v_norm(missile_vel) if v_mag(missile_vel) > 1e-6 else v_norm(rel_pos)
            missile_vel = v_add(missile_vel, v_mul(thrust_dir, motor_accel_ms2 * DT))

            if USE_GRAVITY:
                missile_vel[2] -= GRAVITY * DT

            old_dir_for_turn = v_norm(prev_missile_vel)
            new_dir_for_turn = v_norm(missile_vel)
            turn_dot = max(-1.0, min(1.0, v_dot(old_dir_for_turn, new_dir_for_turn)))
            turn_angle = math.degrees(math.acos(turn_dot))
            actual_turn_rate = turn_angle / DT if DT > 0 else 0.0

            missile_speed = v_mag(missile_vel)
            missile_mach = missile_speed / sound_speed
            missile_alt_km = missile_pos[2] / 1000.0

            base_drag_force = drag_force_newtons(missile_speed, missile_alt_km, missile_diameter_m, drag_coefficient_cd)
            extra_turn_drag_force = turn_drag_force_newtons(base_drag_force, actual_turn_rate, turn_drag_multiplier)
            total_drag_force = base_drag_force + extra_turn_drag_force
            drag_accel_ms2 = total_drag_force / current_mass_kg

            missile_speed = max(0.0, missile_speed - drag_accel_ms2 * DT)
            missile_mach = missile_speed / sound_speed
            missile_vel = v_mul(v_norm(missile_vel), missile_speed)
            missile_pos = v_add(missile_pos, v_mul(missile_vel, DT))

            rel_pos = v_sub(target_pos, missile_pos)
            distance = v_mag(rel_pos)
            closest_distance = closest_distance_between_steps(prev_target_pos, prev_missile_pos, target_pos, missile_pos)

            if notch_started:
                notch_elapsed = time - notch_start_time
                if notch_elapsed < NOTCH_BREAK_TIME:
                    target_phase = "Notch break"
                elif notch_mode == 2 and has_lpi and rwr_ping_visible_timer <= 0:
                    target_phase = "Notch hold"
                else:
                    target_phase = "Notch update"
            else:
                target_phase = "Normal"

            if closest_distance <= PROXY_FUSE_M:
                intercepted = True
                all_hit_times.append(time)
                if activation_time is not None:
                    all_activation_to_hit_times.append(time - activation_time)
                markers["intercepted"] = True
                markers["end_time"] = time
                markers["end_distance"] = closest_distance / 1000.0
                if activation_time is not None:
                    markers["activation_to_intercept_time"] = time - activation_time

                record_point(phase, target_phase, current_stage, current_thrust_n, motor_accel_ms2,
                             actual_target_accel_vec, target_accel_perp_mag, current_mass_kg,
                             remaining_total_fuel, actual_turn_rate, base_drag_force,
                             extra_turn_drag_force, total_drag_force, drag_accel_ms2, current_tti)
                break

            if reached_activation_range:
                ping_timer += DT
                if rwr_ping_visible_timer > 0:
                    rwr_ping_visible_timer = max(0.0, rwr_ping_visible_timer - DT)
                if has_lpi and ping_timer >= 0.5:
                    ping_timer = 0.0
                    ping_chance = lpi_detection_chance(distance, lpi_value)
                    if random.random() < ping_chance:
                        rwr_ping_visible_timer = 0.5
                        if first_ping_distance is None:
                            first_ping_distance = distance / 1000.0
                            first_ping_time = time
                            first_ping_point = (missile_pos[0] / 1000.0, missile_pos[1] / 1000.0, missile_pos[2] / 1000.0)

            record_point(phase, target_phase, current_stage, current_thrust_n, motor_accel_ms2,
                         actual_target_accel_vec, target_accel_perp_mag, current_mass_kg,
                         remaining_total_fuel, actual_turn_rate, base_drag_force,
                         extra_turn_drag_force, total_drag_force, drag_accel_ms2, current_tti)

            time += DT

        if first_ping_distance is not None:
            all_first_ping_distances.append(first_ping_distance)
            all_first_ping_times.append(first_ping_time)
            all_first_ping_points.append(first_ping_point)
            markers["ping_point"] = first_ping_point
            markers["ping_distance"] = first_ping_distance
            markers["ping_time"] = first_ping_time

        if run == 0:
            if not intercepted:
                markers["intercepted"] = False
                if data["distance"]:
                    markers["end_time"] = data["time"][-1]
                    markers["end_distance"] = data["distance"][-1]
                else:
                    markers["end_time"] = time
                    markers["end_distance"] = None
            final_data = {"data": data, "markers": markers}

    avg_ping_distance = None
    avg_ping_time = None
    avg_ping_point = None
    if has_lpi and all_first_ping_distances and runs > 1:
        avg_ping_distance = sum(all_first_ping_distances) / len(all_first_ping_distances)
        avg_ping_time = sum(all_first_ping_times) / len(all_first_ping_times)
        avg_ping_point = (
            sum(p[0] for p in all_first_ping_points) / len(all_first_ping_points),
            sum(p[1] for p in all_first_ping_points) / len(all_first_ping_points),
            sum(p[2] for p in all_first_ping_points) / len(all_first_ping_points),
        )

    avg_activation_to_hit_time = None
    if all_activation_to_hit_times:
        avg_activation_to_hit_time = sum(all_activation_to_hit_times) / len(all_activation_to_hit_times)

    avg_second_pulse_tti = None
    if all_second_pulse_tti:
        avg_second_pulse_tti = sum(all_second_pulse_tti) / len(all_second_pulse_tti)

    if final_data is None:
        final_data = {"data": {}, "markers": {}}

    data = final_data["data"]
    markers = final_data["markers"]

    return {
        "all_hit_times": all_hit_times,
        "all_activation_to_hit_times": all_activation_to_hit_times,
        "avg_activation_to_hit_time": avg_activation_to_hit_time,
        "all_activation_times": all_activation_times,
        "all_activation_distances": all_activation_distances,
        "all_notch_times": all_notch_times,
        "all_angle_change_times": all_angle_change_times,
        "all_second_pulse_times": all_second_pulse_times,
        "avg_second_pulse_tti": avg_second_pulse_tti,

        "final_mx": data.get("mx", []),
        "final_my": data.get("my", []),
        "final_mz": data.get("mz", []),
        "final_tx": data.get("tx", []),
        "final_ty": data.get("ty", []),
        "final_tz": data.get("tz", []),
        "final_time": data.get("time", []),
        "final_missile_mach": data.get("missile_mach", []),
        "final_target_mach": data.get("target_mach", []),
        "final_missile_ms": data.get("missile_ms", []),
        "final_target_ms": data.get("target_ms", []),
        "final_distance": data.get("distance", []),
        "final_phase": data.get("phase", []),
        "final_target_phase": data.get("target_phase", []),
        "final_air_density_factor": data.get("air_density_factor", []),
        "final_air_density_kg_m3": data.get("air_density_kg_m3", []),
        "final_aero_drag_force": data.get("aero_drag_force", []),
        "final_turn_drag_force": data.get("turn_drag_force", []),
        "final_total_drag_force": data.get("total_drag_force", []),
        "final_drag_accel": data.get("drag_accel", []),
        "final_turn_rate": data.get("turn_rate", []),
        "final_thrust": data.get("thrust", []),
        "final_motor_accel": data.get("motor_accel", []),
        "final_target_accel": data.get("target_accel", []),
        "final_target_accel_perp": data.get("target_accel_perp", []),
        "final_current_mass": data.get("current_mass", []),
        "final_remaining_fuel": data.get("remaining_fuel", []),
        "final_tti": data.get("tti", []),

        "final_ping_point": markers.get("ping_point"),
        "final_ping_distance": markers.get("ping_distance"),
        "final_ping_time": markers.get("ping_time"),
        "avg_ping_point": avg_ping_point,
        "avg_ping_distance": avg_ping_distance,
        "avg_ping_time": avg_ping_time,

        "final_activation_point": markers.get("activation_point"),
        "final_activation_time": markers.get("activation_time"),
        "final_activation_to_intercept_time": markers.get("activation_to_intercept_time"),
        "final_notch_point": markers.get("notch_point"),
        "final_angle_change_point": markers.get("angle_change_point"),
        "final_notch_side": markers.get("notch_side"),
        "final_second_pulse_point": markers.get("second_pulse_point"),
        "final_second_pulse_time": markers.get("second_pulse_time"),
        "final_second_pulse_tti": markers.get("second_pulse_tti"),
        "final_intercepted": markers.get("intercepted", False),
        "final_end_time": markers.get("end_time"),
        "final_end_distance": markers.get("end_distance"),

        "total_fuel_kg": total_fuel_kg,
        "dry_mass_kg": dry_mass_kg,
        "initial_first_fuel_kg": initial_first_fuel_kg,
        "initial_second_fuel_kg": initial_second_fuel_kg,
    }


# ----------------------------
# Output
# ----------------------------

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

    if use_dual_pulse:
        if result["all_second_pulse_times"]:
            st.write(f"Second pulse started at: **{sum(result['all_second_pulse_times']) / len(result['all_second_pulse_times']):.2f} sec**")
            if result["avg_second_pulse_tti"] is not None:
                st.write(f"Estimated TTI at second pulse start: **{result['avg_second_pulse_tti']:.2f} sec**")
        else:
            st.write("Second pulse never started.")

    st.write(f"Fuel mass: **{result['total_fuel_kg']:.2f} kg**")
    st.write(f"Dry mass: **{result['dry_mass_kg']:.2f} kg**")
    st.write(f"First pulse fuel: **{result['initial_first_fuel_kg']:.2f} kg**")
    if use_dual_pulse:
        st.write(f"Second pulse fuel: **{result['initial_second_fuel_kg']:.2f} kg**")

    st.write(f"Missile diameter: **{missile_diameter_m:.3f} m**")
    st.write(f"Drag coefficient Cd: **{drag_coefficient_cd:.2f}**")
    st.write(f"Turn drag multiplier: **{turn_drag_multiplier:.2f}x**")

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

    mx = result["final_mx"]
    my = result["final_my"]
    mz = result["final_mz"]
    tx = result["final_tx"]
    ty = result["final_ty"]
    tz = result["final_tz"]

    if mx and tx:
        fig = go.Figure()

        missile_hover = []
        for i in range(len(mx)):
            tti_text = "N/A" if result["final_tti"][i] is None else f"{result['final_tti'][i]:.1f}s"
            missile_hover.append(
                f"Missile<br>"
                f"t: {result['final_time'][i]:.2f}s<br>"
                f"Phase: {result['final_phase'][i]}<br>"
                f"TTI est: {tti_text}<br>"
                f"M: {result['final_missile_mach'][i]:.2f}<br>"
                f"m/s: {result['final_missile_ms'][i]:.0f}<br>"
                f"Mass: {result['final_current_mass'][i]:.1f} kg<br>"
                f"Fuel: {result['final_remaining_fuel'][i]:.1f} kg<br>"
                f"Thrust: {result['final_thrust'][i]:.0f} N<br>"
                f"Motor accel: {result['final_motor_accel'][i]:.1f} m/s²<br>"
                f"G limit: {missile_max_g:.1f}<br>"
                f"Turn: {result['final_turn_rate'][i]:.1f}°/s<br>"
                f"Aero drag: {result['final_aero_drag_force'][i]:.0f} N<br>"
                f"Turn drag: {result['final_turn_drag_force'][i]:.0f} N<br>"
                f"Total drag: {result['final_total_drag_force'][i]:.0f} N<br>"
                f"Drag accel: {result['final_drag_accel'][i]:.1f} m/s²<br>"
                f"Air factor: {result['final_air_density_factor'][i]:.3f}<br>"
                f"Air kg/m³: {result['final_air_density_kg_m3'][i]:.3f}<br>"
                f"Dist: {result['final_distance'][i]:.2f} km<br>"
                f"X/Y/Z: {mx[i]:.2f}, {my[i]:.2f}, {mz[i]:.2f} km"
            )

        target_hover = []
        for i in range(len(tx)):
            target_hover.append(
                f"Target<br>"
                f"t: {result['final_time'][i]:.2f}s<br>"
                f"Phase: {result['final_target_phase'][i]}<br>"
                f"M: {result['final_target_mach'][i]:.2f}<br>"
                f"m/s: {result['final_target_ms'][i]:.0f}<br>"
                f"Heading: {target_heading:.1f}°<br>"
                f"Accel: {result['final_target_accel'][i]:.1f} m/s²<br>"
                f"APN accel: {result['final_target_accel_perp'][i]:.1f} m/s²<br>"
                f"Max M: {target_max_mach:.2f}<br>"
                f"Turn set: {target_turn_rate_deg:.1f}°/s<br>"
                f"Dist: {result['final_distance'][i]:.2f} km<br>"
                f"X/Y/Z: {tx[i]:.2f}, {ty[i]:.2f}, {tz[i]:.2f} km"
            )

        terminal_name = f"APN, N={nav_constant}"
        guidance_name = f"Loft {loft_angle}° + {terminal_name}" if use_loft else terminal_name

        fig.add_trace(go.Scatter3d(x=mx, y=my, z=mz, mode="lines", name="Missile path", line=dict(width=6), hovertext=missile_hover, hoverinfo="text"))
        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines", name="Target path", line=dict(width=6), hovertext=target_hover, hoverinfo="text"))

        launch_hover = (
            f"Missile launch<br>"
            f"Platform M: {launch_platform_mach:.2f}<br>"
            f"Range: {start_horizontal_range:.2f} km<br>"
            f"Alt: {missile_altitude:.2f} km<br>"
            f"Mass: {missile_mass_kg:.1f} kg<br>"
            f"Dry: {result['dry_mass_kg']:.1f} kg<br>"
            f"Fuel: {result['total_fuel_kg']:.1f} kg<br>"
            f"First thrust: {first_pulse_thrust_n:.0f} N<br>"
            f"First burn: {first_pulse_burn_time:.1f}s<br>"
            f"Diameter: {missile_diameter_m:.3f} m<br>"
            f"Cd: {drag_coefficient_cd:.2f}<br>"
            f"Turn drag: {turn_drag_multiplier:.2f}x"
        )

        if use_dual_pulse:
            launch_hover += (
                f"<br>Second thrust: {second_pulse_thrust_n:.0f} N"
                f"<br>Second burn: {second_pulse_burn_time:.1f}s"
                f"<br>Second trigger: {second_pulse_trigger_mode}"
            )
            if second_pulse_trigger_mode == "After flight time":
                launch_hover += f"<br>Trigger time: {second_pulse_trigger_time:.1f}s"
            elif second_pulse_trigger_mode == "At target distance":
                launch_hover += f"<br>Trigger distance: {second_pulse_trigger_distance:.1f} km"
            elif second_pulse_trigger_mode == "Time to target estimate":
                launch_hover += f"<br>Trigger TTI: {second_pulse_trigger_tti:.1f}s"

        if use_manual_launch:
            launch_hover += f"<br>H offset: {launch_horizontal_offset_deg:.1f}°<br>V angle: {launch_vertical_angle_deg:.1f}°"

        fig.add_trace(go.Scatter3d(x=[mx[0]], y=[my[0]], z=[mz[0]], mode="markers+text", name="Missile launch", text=["Launch"], marker=dict(size=6), hovertext=[launch_hover], hoverinfo="text"))

        fig.add_trace(go.Scatter3d(
            x=[tx[0]], y=[ty[0]], z=[tz[0]], mode="markers+text", name="Target start", text=["Target start"], marker=dict(size=6),
            hovertext=[f"Target start<br>Heading: {target_heading:.1f}°<br>M: {target_mach_start:.2f}<br>Max M: {target_max_mach:.2f}<br>Turn: {target_turn_rate_deg:.1f}°/s<br>Alt: {target_altitude:.2f} km"],
            hoverinfo="text",
        ))

        if result["final_activation_point"] is not None:
            p = result["final_activation_point"]
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Seeker activation range marker", text=["Seeker"], marker=dict(size=9), hovertext=[f"Seeker activation<br>Range: {activation_range:.2f} km<br>t: {result['final_activation_time']:.2f}s"], hoverinfo="text"))

        if use_dual_pulse and result["final_second_pulse_point"] is not None:
            p = result["final_second_pulse_point"]
            second_tti_text = "N/A" if result["final_second_pulse_tti"] is None else f"{result['final_second_pulse_tti']:.2f}s"
            fig.add_trace(go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Second pulse start", text=["2nd pulse"], marker=dict(size=9),
                hovertext=[f"Second pulse start<br>t: {result['final_second_pulse_time']:.2f}s<br>TTI est: {second_tti_text}<br>Thrust: {second_pulse_thrust_n:.0f} N<br>Burn: {second_pulse_burn_time:.1f}s<br>Trigger: {second_pulse_trigger_mode}"],
                hoverinfo="text",
            ))

        if result["final_notch_point"] is not None:
            p = result["final_notch_point"]
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Target notch start", text=["Notch"], marker=dict(size=9), hovertext=[f"Target notch<br>Mode: {notch_mode}<br>Side: {result['final_notch_side']}<br>V angle: {notch_vertical_angle:.1f}°<br>Turn: {target_turn_rate_deg:.1f}°/s"], hoverinfo="text"))

        if result["final_angle_change_point"] is not None:
            p = result["final_angle_change_point"]
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Target direction change", text=["Target turn"], marker=dict(size=9), hovertext=[f"Target turn<br>Alt trigger: {change_altitude:.2f} km<br>New heading: {new_target_heading:.1f}°<br>New V: {new_target_climb:.1f}°<br>Turn: {target_turn_rate_deg:.1f}°/s"], hoverinfo="text"))

        if result["final_intercepted"]:
            end_name = "Intercept"
            end_text = "Intercept"
            activation_to_intercept_text = "unknown" if result["final_activation_to_intercept_time"] is None else f"{result['final_activation_to_intercept_time']:.2f}s"
            end_hover = [
                f"Intercept<br>t: {result['final_end_time']:.2f}s<br>Miss: {result['final_end_distance'] * 1000:.2f} m<br>Seeker→hit: {activation_to_intercept_text}<br>Mass: {result['final_current_mass'][-1]:.1f} kg<br>Fuel: {result['final_remaining_fuel'][-1]:.1f} kg"
            ]
        else:
            end_name = "Simulation end"
            end_text = "End"
            final_dist_text = f"{result['final_end_distance']:.2f} km" if result["final_end_distance"] is not None else "unknown"
            end_hover = [
                f"No intercept<br>t: {result['final_end_time']:.2f}s<br>Final dist: {final_dist_text}<br>Mass: {result['final_current_mass'][-1]:.1f} kg<br>Fuel: {result['final_remaining_fuel'][-1]:.1f} kg<br>Target XYZ: {tx[-1]:.2f}, {ty[-1]:.2f}, {tz[-1]:.2f} km<br>Missile XYZ: {mx[-1]:.2f}, {my[-1]:.2f}, {mz[-1]:.2f} km"
            ]

        fig.add_trace(go.Scatter3d(x=[tx[-1]], y=[ty[-1]], z=[tz[-1]], mode="markers+text", name=end_name, text=[end_text], marker=dict(size=9), hovertext=end_hover, hoverinfo="text"))

        if has_lpi and runs == 1 and result["final_ping_point"] is not None:
            p = result["final_ping_point"]
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="First ping", text=["First ping"], marker=dict(size=8), hovertext=[f"First ping<br>Distance: {result['final_ping_distance']:.2f} km<br>t: {result['final_ping_time']:.2f}s"], hoverinfo="text"))

        if has_lpi and runs > 1 and result["avg_ping_point"] is not None:
            p = result["avg_ping_point"]
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name="Average first ping", text=["Avg ping"], marker=dict(size=10), hovertext=[f"Average first ping<br>Distance: {result['avg_ping_distance']:.2f} km<br>t: {result['avg_ping_time']:.2f}s"], hoverinfo="text"))

        fig.update_layout(
            title=f"3D Missile Intercept - {guidance_name}",
            scene=dict(xaxis_title="X km", yaxis_title="Y km", zaxis_title="Altitude km", aspectmode="data"),
            height=750,
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_animation:
            st.subheader("Animated Playback")
            anim_fig = make_animation_figure(result, animation_frame_skip, animation_speed_ms)
            if anim_fig is not None:
                st.plotly_chart(anim_fig, use_container_width=True)
            else:
                st.write("Animation could not be created because there was no path data.")

else:
    st.info("Set the values in the sidebar, then click Run simulation.")
