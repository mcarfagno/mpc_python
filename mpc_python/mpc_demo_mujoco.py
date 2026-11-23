from __future__ import annotations

import pathlib
import signal
import threading
import time

import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer
from cvxpy_mpc import MPC
from cvxpy_mpc.utils import (
    compute_path_from_wp,
    compute_errors,
    detect_obstacle_camera,
    ego_to_global,
    get_ref_trajectory,
    update_path_obstacles,
    obstacle_to_ego,
)

import yaml


# MPC and sim are on 2 threads
# avoids messy global vars
class SharedData:
    """Encapsulates all data shared between the Physics thread and MPC thread."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()

        # Core control & state
        self.state: npt.NDArray[np.float64] = np.zeros(4)
        self.goal_reached: bool = False
        self.is_active: bool = True
        self.mpc_accel: float = 0.0
        self.mpc_steer: float = 0.0

        # Telemetry & visualization
        self.x_mpc_world: npt.NDArray[np.float64] | None = None
        self.mpc_elapsed: float = 0.0

        # Externally detected obstacle (ego frame, set by physics thread)
        self.obstacle: tuple[float, float, float] | None = None


def controller_loop(
    mpc: MPC,
    path: npt.NDArray[np.float64],
    shared: SharedData,
    goal_threshold: float,
    target_speed: float,
) -> None:

    while True:
        start_time = time.time()

        # (safely) grab the latest state from the simulation
        with shared.lock:
            if not shared.is_active or shared.goal_reached:
                break
            current_state = shared.state.copy()  # Global [X, Y, V, Theta]
            global_obs = (
                shared.obstacle
            )  # from external detection pipeline (global frame)
        last_control = (shared.mpc_accel, shared.mpc_steer)
        elapsed = shared.mpc_elapsed

        # Check goal using absolute global coordinates
        goal_dist = np.sqrt(
            (current_state[0] - path[0, -1]) ** 2
            + (current_state[1] - path[1, -1]) ** 2
        )
        if goal_dist < goal_threshold:
            with shared.lock:
                shared.goal_reached = True
            break

        # Add delay compensation
        # ok why we need this in practice? the optimiser takes some time
        # to compute the next command. The mpc should compute the command for t+delay because that is
        # when it will be applied. the actual delay is the expected computation time (assumed from last one)
        pred_state = current_state.copy()
        v = pred_state[2]
        theta = pred_state[3]
        a = last_control[0]
        delta = last_control[1]
        L = mpc.wheelbase

        # Integrate physics forward in global space
        pred_state[0] += v * np.cos(theta) * elapsed
        pred_state[1] += v * np.sin(theta) * elapsed
        pred_state[2] += a * elapsed
        pred_state[3] += (v * np.tan(delta) / L) * elapsed

        # NOTE: we convert the state in ego frame and we use a ego target
        # so we the optimization problem is a bit easier and we save some solver time
        # Get reference trajectory
        target = get_ref_trajectory(
            pred_state, path, target_speed, mpc.control_horizon * mpc.dt, mpc.dt
        )
        pred_ego_state = [0.0, 0.0, pred_state[2], 0.0]

        # Transform global obstacle to ego frame using the same pred_state
        # that the rest of the MPC ego frame is built from
        pred_obstacle = (
            obstacle_to_ego(pred_state, global_obs) if global_obs is not None else None
        )

        x_mpc, u_mpc = mpc.solve(
            pred_ego_state, target, verbose=False, obstacle=pred_obstacle
        )

        # Extract the immediate next optimal control actions
        control = (u_mpc[0, 0], u_mpc[1, 0])
        elapsed = time.time() - start_time

        # Safely push results back to the shared object
        with shared.lock:
            shared.mpc_accel = control[0]
            shared.mpc_steer = control[1]
            shared.mpc_elapsed = elapsed
            shared.x_mpc_world = (
                ego_to_global(pred_state, x_mpc) if x_mpc is not None else None
            )

        # Enforce loop frequency
        elapsed_total = time.time() - start_time
        sleep_time = max(0.0, mpc.dt - elapsed_total)
        time.sleep(sleep_time)


def body_id(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i == -1:
        raise ValueError(f"Body '{name}' not found")
    return i


def get_state(data: mujoco.MjData, bid: int) -> npt.NDArray[np.float64]:
    rot = data.xmat[bid].reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    speed = np.linalg.norm(data.qvel[0:2])
    return np.array([data.xpos[bid][0], data.xpos[bid][1], speed, yaw])

def draw_path(
    viewer: mujoco.viewer.MjViewer,
    path: npt.NDArray[np.float64],
    offset: float = 0.0,
    rgba: npt.NDArray[np.float32] | None = None,
) -> None:
    if rgba is None:
        rgba = np.array([0, 0.6, 1, 1], dtype=np.float32)
    normal_x = -np.sin(path[2, :])
    normal_y = np.cos(path[2, :])
    xs = path[0, :] + offset * normal_x
    ys = path[1, :] + offset * normal_y
    for i in range(path.shape[1] - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        # next geometry slot
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        p1 = np.array([xs[i], ys[i], 0.03], dtype=np.float64)
        p2 = np.array([xs[i + 1], ys[i + 1], 0.03], dtype=np.float64)
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array(
                [0.008, 0.0, 0.0], dtype=np.float64
            ),  # [radius, unused, unused]
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )

        # mujoco handles the vector math to stretch it between p1 and p2
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, p1, p2)

        viewer.user_scn.ngeom += 1


def draw_trail(
    viewer: mujoco.viewer.MjViewer,
    x_hist: list[float],
    y_hist: list[float],
    downsample: int = 10,
) -> None:
    if len(x_hist) < 2:
        return
    for i in range(0, len(x_hist) - 1, downsample):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        alpha = (i + 1) / len(x_hist) * 0.8

        p1 = np.array([x_hist[i], y_hist[i], 0.005], dtype=np.float64)
        p2 = np.array([x_hist[i + 1], y_hist[i + 1], 0.005], dtype=np.float64)

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.02, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([1, 0, 0, alpha], dtype=np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02, p1, p2)
        viewer.user_scn.ngeom += 1


def draw_obstacle(viewer: mujoco.viewer.MjViewer, obstacles) -> None:
    for idx, (ox, oy, rad, _, _) in enumerate(obstacles):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([rad, 0.0, 0.0], dtype=np.float64),
            pos=np.array([ox, oy, 0.0], dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([1, 0, 0, 0.4], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def draw_mpc_preview(
    viewer: mujoco.viewer.MjViewer, x_mpc_world: npt.NDArray[np.float64]
) -> None:
    for i in range(x_mpc_world.shape[1]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        # next geometry slot
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.03, 0.0, 0.0], dtype=np.float64),
            pos=np.array(
                [x_mpc_world[0, i], x_mpc_world[1, i], 0.01], dtype=np.float64
            ),
            mat=np.eye(3).ravel(),
            rgba=np.array([0, 1, 0, 0.6], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def draw_sensor_fov(
    viewer: mujoco.viewer.MjViewer,
    x: float,
    y: float,
    heading: float,
    max_range: float,
    fov_deg: float,
) -> None:
    fov_rad = np.radians(fov_deg)
    right_angle = heading - fov_rad / 2.0
    left_angle = heading + fov_rad / 2.0

    right_end = np.array(
        [x + max_range * np.cos(right_angle), y + max_range * np.sin(right_angle), 0.0]
    )
    left_end = np.array(
        [x + max_range * np.cos(left_angle), y + max_range * np.sin(left_angle), 0.0]
    )
    origin = np.array([x, y, 0.0])

    rgba = np.array([0.8, 0.8, 0, 0.2], dtype=np.float32)

    for end in (right_end, left_end):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.02, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01, origin, end)
        viewer.user_scn.ngeom += 1

    num_arc_pts = 20
    prev = None
    for i in range(num_arc_pts + 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        t = i / num_arc_pts
        angle = right_angle + t * fov_rad
        p = np.array(
            [x + max_range * np.cos(angle), y + max_range * np.sin(angle), 0.0]
        )
        if prev is not None:
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.01, 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3).ravel(),
                rgba=rgba,
            )
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01, prev, p)
            viewer.user_scn.ngeom += 1
        prev = p


# here we run the sim loop
def main() -> None:
    model_path = pathlib.Path(__file__).parent / "models" / "mushr" / "mush_nano.xml"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    bid = body_id(m, "buddy")

    sim_config = yaml.safe_load(
        (pathlib.Path(__file__).parent / "config" / "simulation.yaml").read_text()
    )
    start = sim_config["start"]
    target_speed = sim_config["target_speed"]
    sensor_max_range = sim_config["sensor"]["max_range"]
    sensor_fov_deg = sim_config["sensor"]["fov_deg"]
    goal_threshold = sim_config["goal_threshold"]

    d.qpos[:4] = [start["x"], start["y"], 0.1, 1.0]
    mujoco.mj_forward(m, d)

    path = compute_path_from_wp(
        sim_config["path"]["waypoints_x"],
        sim_config["path"]["waypoints_y"],
        sim_config["path"]["interpolation_step"],
    )

    path_obstacles = list(sim_config["obstacles"])

    # move the obstacles along the path and computes the x and y velocity
    # [x,y,r,vx,vy] just as they would come out from a tracker (e.g. ekf estimate)
    dynamic_obstacle_list = update_path_obstacles(path_obstacles, path, 0.0)
    mpc = MPC(
        "config/mpc.yaml",
        # overriedes the default yaml
        horizon_time=4.0,
        state_cost=[1.0, 50.0, 10.0, 20.0],
        final_state_cost=[1.0, 50.0, 10.0, 20.0],
    )

    steer_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "buddy_steering_wheel")
    steer_qaddr = m.jnt_qposadr[steer_jnt]

    shared = SharedData()
    mpc_thread = threading.Thread(
        target=controller_loop,
        args=(mpc, path, shared, goal_threshold, target_speed),
        daemon=True,
    )

    # handle CTRL-C for the mpc thread
    shutdown_flag = threading.Event()

    def handle_shutdown(signum, frame):
        shutdown_flag.set()

    signal.signal(signal.SIGINT, handle_shutdown)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -45

        fps = 60.0
        render_dt = 1.0 / fps

        x_hist = []
        y_hist = []
        cte_hist = []
        heading_error_hist = []
        cte_rmse = -1
        heading_rmse = -1

        sim_start_time = time.perf_counter()
        mpc_thread.start()

        try:

            while viewer.is_running() and not shutdown_flag.is_set():

                # Check for completion
                if shared.goal_reached:
                    viewer.set_texts(
                        [
                            (
                                None,
                                None,
                                f"GOAL REACHED\n"
                                f"Final RMSE:   CTE {cte_rmse:.3f} m  |  heading {heading_rmse:.1f} deg\n",
                                "",
                            )
                        ]
                    )
                    viewer.sync()
                    print(
                        "\nGoal reached! Close the viewer window or press CTRL-C to exit."
                    )
                    # Idle until the user closes the window or forces a KeyboardInterrupt
                    while viewer.is_running():
                        time.sleep(0.1)
                    break

                elapsed_real_time = time.perf_counter() - sim_start_time

                current_state = get_state(d, bid)

                # External obstacle detection pipeline (global frame)
                if path_obstacles:
                    detected_obs = detect_obstacle_camera(
                        dynamic_obstacle_list,
                        current_state[0],
                        current_state[1],
                        current_state[3],
                        sensor_max_range,
                        sensor_fov_deg,
                    )
                else:
                    detected_obs = None

                # Sync with MPC Thread
                with shared.lock:
                    shared.state[:] = current_state
                    shared.obstacle = detected_obs
                    mpc_elapsed = shared.mpc_elapsed
                    mpc_accel = shared.mpc_accel
                    mpc_steer = shared.mpc_steer
                    x_mpc_world = shared.x_mpc_world

                # Log position etc...
                x_hist.append(current_state[0])
                y_hist.append(current_state[1])
                cte, heading_err = compute_errors(current_state, path)
                cte_hist.append(cte)
                heading_error_hist.append(np.degrees(heading_err))
                cte_rmse = np.sqrt(np.mean(np.square(cte_hist)))
                heading_rmse = np.sqrt(np.mean(np.square(heading_error_hist)))

                # Step physics
                while d.time < elapsed_real_time and not shutdown_flag.is_set():
                    # ZERO-ORDER HOLD (ZOH)
                    # The MPC outputs a target steering angle.
                    # Since the low-level actuator (aka steering wheel PID) usually
                    # tracks position directly, we treat this command as a constant step.
                    # We hold it flat (Zero-Order Hold) across the entire window.
                    d.ctrl[0] = mpc_steer

                    # FIRST-ORDER HOLD (FOH)
                    # MuJoCo's wheel actuators expect a velocity command,
                    # but the MPC outputs an acceleration.
                    #
                    # If we applied a raw step jump to velocity (speed+=mpc_acc*DT), it would imply infinite acceleration.
                    # Instead, we integrate the acceleration command over every microscopic physics step. This smoothly ramps the
                    # velocity command over time, creating a First-Order Hold that perfectly mimics
                    # a real motor.
                    d.ctrl[1] += mpc_accel * m.opt.timestep

                    dynamic_obstacle_list[:] = update_path_obstacles(
                        path_obstacles, path, m.opt.timestep
                    )
                    mujoco.mj_step(m, d)

                # Update camera position to follow the car
                viewer.cam.lookat[:] = [current_state[0], current_state[1], 0.0]

                # re-draw markers
                viewer.user_scn.ngeom = 0
                draw_path(viewer, path)
                if mpc.track_half_width > 0.0:
                    inner = max(mpc.track_half_width - mpc._vehicle_buffer, 0.0)
                    bound = np.array([0.45, 0.45, 0.45, 1.0], dtype=np.float32)
                    draw_path(viewer, path, inner, bound)
                    draw_path(viewer, path, -inner, bound)
                if path_obstacles:
                    draw_obstacle(viewer, dynamic_obstacle_list)
                    draw_sensor_fov(
                        viewer,
                        current_state[0],
                        current_state[1],
                        current_state[3],
                        sensor_max_range,
                        sensor_fov_deg,
                    )
                draw_trail(viewer, x_hist, y_hist)
                if x_mpc_world is not None:
                    draw_mpc_preview(viewer, x_mpc_world)

                # Update the HUD
                actual_steer = np.degrees(d.qpos[steer_qaddr])
                goal_dist = np.hypot(
                    current_state[0] - path[0, -1], current_state[1] - path[1, -1]
                )

                viewer.set_texts(
                    [
                        (
                            None,
                            None,
                            f"MPC Demo\n"
                            f"state:  v {current_state[2]:.2f} m/s  |  steer {actual_steer:.1f} deg\n"
                            f"MPC:    accel {mpc_accel:.2f} m/s2  |  steer {np.degrees(mpc_steer):.1f} deg  |  {mpc_elapsed*1000:.0f} ms\n"
                            f"error:  CTE {cte:.3f} m  |  heading {np.degrees(heading_err):.1f} deg\n"
                            f"RMSE:   CTE {cte_rmse:.3f} m  |  heading {heading_rmse:.1f} deg\n"
                            f"goal:   {goal_dist:.2f} m\n"
                            f"avoid:  {'YES' if detected_obs is not None else 'no' if path_obstacles else 'off'}\n",
                            "",
                        )
                    ]
                )
                viewer.sync()

                # Frame limiting (sleep just enough to hit 60 FPS)
                time_until_next_frame = render_dt - (
                    time.perf_counter() - elapsed_real_time - sim_start_time
                )
                if time_until_next_frame > 0:
                    time.sleep(time_until_next_frame)
        except Exception as e:
            print(e)

        finally:
            shared.is_active = False
            mpc_thread.join(timeout=1.0)
            if mpc_thread.is_alive():
                print("MPC thread is still alive...")

            viewer.clear_texts()


if __name__ == "__main__":
    main()
