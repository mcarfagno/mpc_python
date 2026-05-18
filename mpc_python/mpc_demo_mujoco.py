from __future__ import annotations

import pathlib
import threading
import time

import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer
from cvxpy_mpc import MPC, VehicleModel
from cvxpy_mpc.utils import (
    compute_path_from_wp,
    compute_errors,
    ego_to_global,
    get_ref_trajectory,
)

TARGET_VEL = 1.0
T = 4.0
DT = 0.2  # controller time step


# MPC and sim are on 2 threads
# avoids messy global vars
class SharedData:
    """Encapsulates all data shared between the Physics thread and MPC thread."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()

        # Core control & state
        self.state: npt.NDArray[np.float64] = np.zeros(5)
        self.goal_reached: bool = False
        self.is_active: bool = True
        self.mpc_accel: float = 0.0
        self.mpc_steer_rate: float = 0.0

        # Telemetry & visualization
        self.x_mpc_world: npt.NDArray[np.float64] | None = None
        self.mpc_elapsed: float = 0.0


def controller_loop(
    mpc: MPC, path: npt.NDArray[np.float64], shared: SharedData
) -> None:

    while True:  # steer is usually zero
        start_time = time.time()

        # (safely) grab the latest state from the simulation
        with shared.lock:
            if not shared.is_active or shared.goal_reached:
                break
            current_state = shared.state.copy()  # Global [X, Y, V, Theta, Delta]
            last_control = (shared.mpc_accel, shared.mpc_steer_rate)
            elapsed = shared.mpc_elapsed

        # Check goal using absolute global coordinates
        goal_dist = np.sqrt(
            (current_state[0] - path[0, -1]) ** 2
            + (current_state[1] - path[1, -1]) ** 2
        )
        if goal_dist < 0.2:
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
        delta = pred_state[4]
        a = last_control[0]
        delta_dot = last_control[1]
        L = mpc.vehicle.wheelbase

        # Integrate physics forward in global space
        # NOTE: this is just for reference lookup
        pred_state[0] += v * np.cos(theta) * elapsed
        pred_state[1] += v * np.sin(theta) * elapsed
        pred_state[2] += a * elapsed
        pred_state[3] += (v * np.tan(delta) / L) * elapsed
        pred_state[4] += delta_dot * elapsed

        # NOTE: we convert the state in ego frame and we use a ego target
        # so we the optimization problem is a bit easier and we save some solver time
        # Get reference trajectory
        target = get_ref_trajectory(pred_state, path, TARGET_VEL, T, DT)

        # MPC initial state in ego frame of pred_state
        pred_ego_state = np.array(
            [
                0.0,
                0.0,
                pred_state[2],
                0.0,
                pred_state[4],
            ]
        )

        x_mpc, u_mpc = mpc.solve(pred_ego_state, target, verbose=False)

        # Extract the immediate next optimal control actions
        control = (u_mpc[0, 0], u_mpc[1, 0])
        elapsed = time.time() - start_time

        # Safely push results back to the shared object
        with shared.lock:
            shared.mpc_accel = control[0]
            shared.mpc_steer_rate = control[1]
            shared.mpc_elapsed = elapsed
            shared.x_mpc_world = (
                ego_to_global(pred_state, x_mpc) if x_mpc is not None else None
            )

        # Enforce loop frequency
        elapsed_total = time.time() - start_time
        sleep_time = max(0.0, DT - elapsed_total)
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


def draw_path(viewer: mujoco.viewer.MjViewer, path: npt.NDArray[np.float64]) -> None:
    for i in range(path.shape[1] - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        # next geometry slot
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        p1 = np.array([path[0, i], path[1, i], 0.03], dtype=np.float64)
        p2 = np.array([path[0, i + 1], path[1, i + 1], 0.03], dtype=np.float64)

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array(
                [0.008, 0.0, 0.0], dtype=np.float64
            ),  # [radius, unused, unused]
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([0, 0.6, 1, 1], dtype=np.float32),
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


# here we run the sim loop
def main() -> None:
    model_path = pathlib.Path(__file__).parent / "models" / "mushr" / "mush_nano.xml"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    bid = body_id(m, "buddy")

    d.qpos[:4] = [0.0, 0.3, 0.1, 1.0]
    mujoco.mj_forward(m, d)

    path = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, 3, -1, -6, -2, -2],
        0.05,
    )

    state_cost = [
        1.0,
        80.0,
        10.0,
        20.0,
        1.0,
    ]  # [Along-track, Cross-track, Velocity, Heading, Steer]
    actuation_cost = [10.0, 10.0]

    mpc = MPC(
        VehicleModel(),
        T,
        DT,
        state_cost,
        state_cost,
        actuation_cost,
        actuation_cost,
    )

    steer_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "buddy_steering_wheel")
    steer_qaddr = m.jnt_qposadr[steer_jnt]

    shared = SharedData()
    mpc_thread = threading.Thread(
        target=controller_loop, args=(mpc, path, shared), daemon=True
    )

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -45

        fps = 60.0
        render_dt = 1.0 / fps

        control = [0.0, 0.0]  # steer and speed
        x_hist = []
        y_hist = []
        cte_hist = []
        heading_error_hist = []
        cte_rmse = -1
        heading_rmse = -1

        sim_start_time = time.perf_counter()

        try:
            input("\033[92mPress Enter to continue...\033[0m")
            mpc_thread.start()

            while viewer.is_running():

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

                # Step physics
                while d.time < elapsed_real_time:
                    d.ctrl[:] = control
                    mujoco.mj_step(m, d)

                # TODO: add steer to get_state
                actual_steer = d.qpos[steer_qaddr]  # Radians
                current_state = np.append(get_state(d, bid), actual_steer)

                # Sync with MPC Thread
                with shared.lock:
                    shared.state[:] = current_state
                    mpc_elapsed = shared.mpc_elapsed
                    mpc_accel = shared.mpc_accel
                    mpc_steer_rate = shared.mpc_steer_rate
                    x_mpc_world = shared.x_mpc_world

                # Log position etc...
                x_hist.append(current_state[0])
                y_hist.append(current_state[1])
                cte, heading_err = compute_errors(current_state, path)
                cte_hist.append(cte)
                heading_error_hist.append(np.degrees(heading_err))
                cte_rmse = np.sqrt(np.mean(np.square(cte_hist)))
                heading_rmse = np.sqrt(np.mean(np.square(heading_error_hist)))

                # Update Zero-Order Hold control
                control[0] = current_state[4] + mpc_steer_rate * DT
                control[1] = current_state[2] + mpc_accel * DT

                # Update camera position to follow the car
                viewer.cam.lookat[:] = [current_state[0], current_state[1], 0.0]

                # re-draw markers
                viewer.user_scn.ngeom = 0
                draw_path(viewer, path)
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
                            f"MPC:    accel {mpc_accel:.2f} m/s2  |  steer speed {np.degrees(mpc_steer_rate):.1f} deg  |  {mpc_elapsed*1000:.0f} ms\n"
                            f"error:  CTE {cte:.3f} m  |  heading {np.degrees(heading_err):.1f} deg\n"
                            f"RMSE:   CTE {cte_rmse:.3f} m  |  heading {heading_rmse:.1f} deg\n"
                            f"goal:   {goal_dist:.2f} m\n",
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

        except KeyboardInterrupt:
            print("\nInterrupted by user (CTRL-C). Shutting down...")

        finally:
            with shared.lock:
                shared.is_active = False
            viewer.clear_texts()


if __name__ == "__main__":
    main()
