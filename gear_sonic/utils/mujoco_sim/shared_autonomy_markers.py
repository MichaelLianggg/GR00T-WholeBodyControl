"""Draw the shared-autonomy object and grasp targets in the MuJoCo viewer.

The C++ deploy binary declares the object pose and the two grasp targets in a
small YAML file, expressed in the **pelvis frame** (see
``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/shared_autonomy.hpp``).
MuJoCo draws in the world frame, so every marker is transformed each frame:

    p_world = p_pelvis_world + R(q_pelvis_world) @ p_target_pelvis

Watching the markers ride along with the robot is the point: an object that is
constant in the pelvis frame is fixed *relative to the robot*, not to the room.
Seeing that move makes the limitation obvious in a way a comment cannot.

This module only reads the config; it never writes to it and never touches the
simulation state. If the file is missing, malformed, or the installed MuJoCo is
too old for custom viewer geometry, drawing is silently skipped and the
simulator runs exactly as before.
"""

from __future__ import annotations

import json
import logging
import pathlib
import struct

import numpy as np

try:
    import msgpack
except ImportError:  # status overlay is optional
    msgpack = None

try:
    import zmq
except ImportError:  # publishing is optional
    zmq = None

logger = logging.getLogger(__name__)

try:
    import mujoco
except ImportError:  # pragma: no cover - the simulator would not run at all
    mujoco = None


# Marker colours (RGBA).
_OBJECT_RGBA = (1.0, 0.85, 0.1, 0.45)  # translucent yellow box
_LEFT_TARGET_RGBA = (0.1, 0.9, 0.2, 0.9)  # green
_RIGHT_TARGET_RGBA = (0.2, 0.5, 1.0, 0.9)  # blue
_LEFT_WRIST_RGBA = (0.1, 0.9, 0.2, 0.75)  # green, wrist->target error rod
_RIGHT_WRIST_RGBA = (0.2, 0.5, 1.0, 0.75)  # blue, wrist->target error rod

_TARGET_RADIUS = 0.025
_WRIST_RADIUS = 0.02
_ERROR_ROD_RADIUS = 0.006
_BEACON_RADIUS = 0.05
_BEACON_HEIGHT = 0.45  # above the pelvis, so it stays visible from any angle

#: Task states, mirroring SharedAutonomyWrapper::TaskState. The colour IS the
#: readout -- the passive MuJoCo viewer has no text overlay API, so state is
#: shown by tinting the object and a beacon above the robot.
_STATE_COLOURS = {
    0: ((0.55, 0.55, 0.58), "MANUAL"),  # grey   - SONIC alone
    1: ((1.00, 0.85, 0.10), "ALIGN"),   # yellow - assist pulling in
    2: ((1.00, 0.45, 0.05), "GRASP"),   # orange - hands closing
    3: ((0.10, 0.85, 0.25), "LIFT"),    # green  - holding and lifting
    4: ((0.90, 0.10, 0.10), "ABORT"),   # red    - latched fault
}
_UNKNOWN_COLOUR = (0.4, 0.4, 0.45)


def load_object_config(path: str | pathlib.Path) -> dict | None:
    """Parse the shared-autonomy YAML the way the C++ side does.

    Deliberately a small hand-rolled reader rather than PyYAML: it must accept
    exactly the keys the deploy binary accepts, and the simulator's virtualenv
    should not grow a dependency just to draw three spheres.

    Returns ``None`` (with a warning) if the file cannot be used.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        logger.warning("[SharedAutonomy] marker config not found: %s", path)
        return None

    position = None
    quaternion = None
    size = None
    left_pos = None
    right_pos = None

    try:
        for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.split("#", 1)[0]
            if ":" not in line:
                if line.strip():
                    raise ValueError(f"line {line_no}: expected 'key: value'")
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            fields = value.split()

            if key == "object_frame":
                if fields and fields[0] != "pelvis":
                    raise ValueError(
                        f"line {line_no}: object_frame must be 'pelvis', got '{fields[0]}'"
                    )
            elif key == "object_position":
                position = [float(v) for v in fields[:3]]
            elif key == "object_quaternion":
                quaternion = [float(v) for v in fields[:4]]
            elif key == "object_size":
                size = [float(v) for v in fields[:3]]
            elif key == "left_grasp_position":
                left_pos = [float(v) for v in fields[:3]]
            elif key == "right_grasp_position":
                right_pos = [float(v) for v in fields[:3]]
            # Other keys (assist tunables, grasp quaternions) do not affect drawing.
    except (ValueError, IndexError) as exc:
        logger.warning("[SharedAutonomy] cannot parse %s: %s", path, exc)
        return None

    if position is None or quaternion is None or len(position) != 3 or len(quaternion) != 4:
        logger.warning(
            "[SharedAutonomy] %s lacks a usable object_position / object_quaternion", path
        )
        return None

    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        logger.warning("[SharedAutonomy] %s: object_quaternion has zero norm", path)
        return None
    quaternion = [c / norm for c in quaternion]

    # Same defaulting rule as the C++ parser: box side centres when explicit
    # grasp positions are absent.
    if left_pos is None and right_pos is None and size is not None and len(size) == 3:
        left_pos = [0.0, 0.5 * size[1], 0.0]
        right_pos = [0.0, -0.5 * size[1], 0.0]

    return {
        "position": np.asarray(position, dtype=float),
        "quaternion": np.asarray(quaternion, dtype=float),  # wxyz
        "size": np.asarray(size, dtype=float) if size is not None and len(size) == 3 else None,
        "left_grasp": np.asarray(left_pos, dtype=float) if left_pos is not None else None,
        "right_grasp": np.asarray(right_pos, dtype=float) if right_pos is not None else None,
    }


class SharedAutonomyMarkers:
    """Per-frame viewer overlay for the object and the two grasp targets.

    Also renders the robot's actual wrist positions (faint, same colours) so the
    target/actual gap is visible directly, and can print the wrists in pelvis
    frame for cross-checking against the C++ side's ``shared_autonomy.csv``.
    """

    #: Wrist bodies, matching the ones the C++ FK uses for the grasp error.
    WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    #: Optional physical object in the scene. Present in scene_43dof_pick.xml.
    OBJECT_BODY = "sa_object"

    #: Wire format shared with the C++ subscriber (see zmq_packed_message_subscriber.hpp).
    HEADER_SIZE = 1280
    TOPIC = b"sa_object"

    def __init__(
        self,
        config_path: str,
        report_period_s: float = 0.0,
        anchor_to_world: bool = False,
        publish_port: int = 0,
    ):
        self.config = load_object_config(config_path) if config_path else None
        self.report_period_s = report_period_s
        self.anchor_to_world = anchor_to_world
        self._last_report = 0.0
        self._warned_unsupported = False
        self._resolved = False
        self._pelvis_id = -1
        self._wrist_ids: list[int] = []
        self._object_body_id = -1

        # World-frame anchor, captured on the first frame once the robot has
        # settled into the scene. Until then the object rides the pelvis.
        self._anchor_pos: np.ndarray | None = None
        self._anchor_mat: np.ndarray | None = None

        self.status: dict = {}
        self._status_socket = None
        self._last_state_name = None
        self._status_silent_frames = 0
        self._warned_no_status = False

        self._socket = None
        if publish_port > 0 and self.active:
            if zmq is None:
                logger.warning("[SharedAutonomy] pyzmq not installed; object pose not published")
            else:
                ctx = zmq.Context.instance()
                self._socket = ctx.socket(zmq.PUB)
                self._socket.bind(f"tcp://*:{publish_port}")
                print(
                    f"[SharedAutonomy][sim] publishing object pose on tcp://*:{publish_port} "
                    f"topic='sa_object'"
                )

    def connect_status(self, host: str, port: int) -> None:
        """Subscribe to the deploy binary's task-status telemetry."""
        if port <= 0:
            return
        if zmq is None or msgpack is None:
            logger.warning(
                "[SharedAutonomy] pyzmq/msgpack missing; task state will not be shown"
            )
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"sa_status")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 0)
        sock.connect(f"tcp://{host}:{port}")
        self._status_socket = sock
        print(
            f"[SharedAutonomy][sim] task state overlay: subscribed to tcp://{host}:{port}\n"
            "                      colours: MANUAL=grey  ALIGN=yellow  GRASP=orange  "
            "LIFT=green  ABORT=red"
        )

    def _poll_status(self) -> None:
        """Drain the status socket. Never blocks; a silent publisher is fine."""
        if self._status_socket is None:
            return
        try:
            while True:
                raw = self._status_socket.recv(zmq.NOBLOCK)
                payload = raw[len(b"sa_status") :]
                self.status = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception:
            pass  # zmq.Again, or a malformed frame -- keep the previous status

        name = self.status.get("state_name")
        if name is not None and name != self._last_state_name:
            self._last_state_name = name
            print(f"[SharedAutonomy][sim] task state -> {name}")

        # Subscribed but nothing arriving is the common failure, and it is
        # invisible in the viewer (markers keep their default colours). Say so
        # once rather than leaving the user to guess which end is at fault.
        if not self.status and not self._warned_no_status:
            self._status_silent_frames += 1
            if self._status_silent_frames > 500:
                self._warned_no_status = True
                print(
                    "[SharedAutonomy][sim] no task status received yet. Check that the deploy "
                    "binary was started with --enable-shared-autonomy AND --task-status-port "
                    "matching this port, and that control has been started (press ]).",
                )

    def _state_colour(self) -> tuple:
        return _STATE_COLOURS.get(int(self.status.get("state", -1)), (_UNKNOWN_COLOUR, "?"))[0]

    def anchor_here(self) -> None:
        """Re-anchor the object at wherever it currently appears in the world."""
        self._anchor_pos = None
        self._anchor_mat = None
        if self.anchor_to_world:
            print("[SharedAutonomy][sim] object will re-anchor on the next frame")

    @property
    def active(self) -> bool:
        return self.config is not None

    def _resolve_bodies(self, model) -> None:
        if self._resolved:
            return
        self._resolved = True
        try:
            self._pelvis_id = model.body("pelvis").id
        except KeyError:
            logger.warning("[SharedAutonomy] no 'pelvis' body; markers disabled")
            self.config = None
            return
        # A scene may carry a real, physical object (scene_43dof_pick.xml). When
        # it does, its simulated pose supersedes the static config pose: the
        # simulator then acts as a genuine perception source rather than echoing
        # the config back, and the box that is drawn is the box that can be hit.
        try:
            self._object_body_id = model.body(self.OBJECT_BODY).id
            print(
                f"[SharedAutonomy][sim] found physical body '{self.OBJECT_BODY}'; "
                "its simulated pose overrides the config pose"
            )
        except KeyError:
            self._object_body_id = -1

        for name in self.WRIST_BODIES:
            try:
                self._wrist_ids.append(model.body(name).id)
            except KeyError:
                logger.warning("[SharedAutonomy] sim model has no '%s'", name)
                self._wrist_ids.append(-1)

    @staticmethod
    def _quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
        mat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(mat, np.ascontiguousarray(quat_wxyz, dtype=float))
        return mat.reshape(3, 3)

    def _add(self, scene, geom_type, size, pos, mat, rgba) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            np.asarray(size, dtype=float),
            np.asarray(pos, dtype=float),
            np.asarray(mat, dtype=float).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def draw(self, viewer, model, data) -> None:
        """Refresh the overlay. Safe to call every frame; never raises."""
        if not self.active or viewer is None or mujoco is None:
            return

        scene = getattr(viewer, "user_scn", None)
        if scene is None:
            if not self._warned_unsupported:
                self._warned_unsupported = True
                logger.warning(
                    "[SharedAutonomy] this MuJoCo build has no viewer.user_scn "
                    "(needs MuJoCo >= 3.0); markers disabled"
                )
            return

        try:
            self._resolve_bodies(model)
            if not self.active:
                return

            self._poll_status()
            state = int(self.status.get("state", -1))
            state_rgb = self._state_colour()
            grasping = bool(self.status.get("grasp_trigger", 0))
            lifting = bool(self.status.get("lift_trigger", 0))

            pelvis_pos = np.array(data.xpos[self._pelvis_id], dtype=float)
            pelvis_mat = np.array(data.xmat[self._pelvis_id], dtype=float).reshape(3, 3)

            def to_world(p_pelvis: np.ndarray) -> np.ndarray:
                return pelvis_pos + pelvis_mat @ p_pelvis

            scene.ngeom = 0

            # Resolve the object pose for this frame.
            #
            # Default: the config pose is pelvis-relative, so the object rides
            # the robot -- which is the honest depiction of what the deploy
            # binary actually targets, since it has no odometry.
            #
            # anchor_to_world: latch the object's world pose on the first frame
            # and hold it there. The pelvis-frame pose is then recomputed each
            # tick and published, so the C++ side tracks a room-fixed object
            # without needing odometry of its own. Simulation only -- a real
            # robot has no such ground truth.
            obj_rel_mat = self._quat_to_mat(self.config["quaternion"])
            if self._object_body_id >= 0:
                # Real body in the scene: read its actual pose. No anchoring
                # needed -- physics already keeps it where it belongs.
                obj_world = np.array(data.xpos[self._object_body_id], dtype=float)
                obj_mat = np.array(data.xmat[self._object_body_id], dtype=float).reshape(3, 3)
                obj_pelvis_pos = pelvis_mat.T @ (obj_world - pelvis_pos)
                obj_pelvis_mat = pelvis_mat.T @ obj_mat
            elif self.anchor_to_world:
                if self._anchor_pos is None:
                    self._anchor_pos = to_world(self.config["position"])
                    self._anchor_mat = pelvis_mat @ obj_rel_mat
                    print(
                        "[SharedAutonomy][sim] object anchored in world at "
                        f"({self._anchor_pos[0]:+.3f}, {self._anchor_pos[1]:+.3f}, "
                        f"{self._anchor_pos[2]:+.3f})"
                    )
                obj_world = self._anchor_pos
                obj_mat = self._anchor_mat
                # Back-project into the current pelvis frame for the publisher.
                obj_pelvis_pos = pelvis_mat.T @ (obj_world - pelvis_pos)
                obj_pelvis_mat = pelvis_mat.T @ obj_mat
            else:
                obj_pelvis_pos = self.config["position"]
                obj_pelvis_mat = obj_rel_mat
                obj_world = to_world(obj_pelvis_pos)
                obj_mat = pelvis_mat @ obj_rel_mat

            self._publish(obj_pelvis_pos, obj_pelvis_mat)
            size = self.config["size"]
            # The object carries the task colour, so state is readable straight
            # from the scene without a text overlay (the passive viewer has none).
            obj_rgba = (*state_rgb, 0.45) if state >= 0 else _OBJECT_RGBA
            if size is not None:
                self._add(
                    scene, mujoco.mjtGeom.mjGEOM_BOX, 0.5 * size, obj_world, obj_mat, obj_rgba
                )
            else:
                self._add(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    (0.03, 0.0, 0.0),
                    obj_world,
                    np.eye(3),
                    obj_rgba,
                )

            # Grasp targets, in the object's own orientation.
            for key, rgba in (
                ("left_grasp", _LEFT_TARGET_RGBA),
                ("right_grasp", _RIGHT_TARGET_RGBA),
            ):
                rel = self.config[key]
                if rel is None:
                    continue
                # Same composition the C++ side does: T_target = T_object * T_rel.
                radius = _TARGET_RADIUS * (1.6 if grasping else 1.0)
                self._add(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    (radius, 0.0, 0.0),
                    obj_world + obj_mat @ rel,
                    np.eye(3),
                    rgba,
                )

            # Wrist-to-target error, drawn as a rod rather than a marker at the
            # wrist itself: the wrist body origin sits inside the robot's own
            # hand mesh (its geoms have a 0.058 m bounding radius), so a small
            # sphere there is completely buried and invisible.  The rod starts
            # inside the hand but most of it is in free space, and its length
            # IS the position error the console reports as GraspErr.
            for body_id, key, rgba in zip(
                self._wrist_ids,
                ("left_grasp", "right_grasp"),
                (_LEFT_WRIST_RGBA, _RIGHT_WRIST_RGBA),
                strict=False,
            ):
                if body_id < 0:
                    continue
                wrist = np.array(data.xpos[body_id], dtype=float)
                rel = self.config[key]
                if rel is None:
                    continue
                target = obj_world + obj_mat @ rel
                if float(np.linalg.norm(target - wrist)) < 1e-4:
                    continue
                self._add_connector(scene, wrist, target, _ERROR_ROD_RADIUS, rgba)

            # Beacon above the robot: always in view, even looking away from
            # the object, and the quickest way to read the state while driving.
            if state >= 0:
                self._add(
                    scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    (_BEACON_RADIUS, 0.0, 0.0),
                    pelvis_pos + np.array([0.0, 0.0, _BEACON_HEIGHT]),
                    np.eye(3),
                    (*state_rgb, 0.9),
                )

            # During LIFT, show how far the target has been raised.
            if lifting:
                offset = float(self.status.get("lift_offset_m", 0.0))
                if offset > 1e-4:
                    # The deploy side raises the grasp target along PELVIS +Z, so
                    # the indicator is vertical in the world, not along the
                    # object's own Z -- with the crate yawed, object -Z points
                    # sideways and the rod would lie flat.
                    base = obj_world - np.array([0.0, 0.0, offset])
                    self._add_connector(scene, base, obj_world, 0.008, (*state_rgb, 0.8))

            self._maybe_report(data, pelvis_pos, pelvis_mat)
        except Exception as exc:  # never let an overlay break the simulator
            logger.warning("[SharedAutonomy] marker drawing disabled after error: %s", exc)
            self.config = None

    def _add_connector(self, scene, start, end, width, rgba) -> None:
        """Draw a rod between two points.

        mjv_initGeom cannot express this: for a capsule it broadcasts size[0]
        across all three components, so the half-length is silently lost and
        every rod comes out as a tiny ball.  mjv_connector is the purpose-built
        call and sets pos/mat/size from the endpoints.
        """
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(width),
            np.asarray(start, dtype=float),
            np.asarray(end, dtype=float),
        )
        scene.ngeom += 1

    @staticmethod
    def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat, dtype=float).flatten())
        return quat

    def _publish(self, pos_pelvis: np.ndarray, mat_pelvis: np.ndarray) -> None:
        """Send the object pose, in the pelvis frame, to the deploy binary.

        Pelvis frame is deliberate: the receiver has no odometry, so anything
        world-referenced would be unusable there. A real perception node would
        publish this same message after projecting its own detection through
        the robot's kinematics.
        """
        if self._socket is None:
            return
        quat = self._mat_to_quat(mat_pelvis)
        fields = [
            {"name": "object_position", "dtype": "f64", "shape": [3]},
            {"name": "object_quaternion", "dtype": "f64", "shape": [4]},
        ]
        header = json.dumps(
            {"v": 1, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")
        ).encode("utf-8")
        payload = struct.pack("<3d", *pos_pelvis) + struct.pack("<4d", *quat)
        try:
            self._socket.send(self.TOPIC + header.ljust(self.HEADER_SIZE, b"\x00") + payload)
        except Exception as exc:  # a dead subscriber must not stall the sim
            logger.warning("[SharedAutonomy] object pose publish failed: %s", exc)
            self._socket = None

    def _maybe_report(self, data, pelvis_pos: np.ndarray, pelvis_mat: np.ndarray) -> None:
        """Print the wrists in pelvis frame, for cross-checking the C++ CSV.

        These should match ``left_wrist_x/y/z`` and ``right_wrist_x/y/z`` in
        ``shared_autonomy.csv`` to within the two models' differences. They are
        computed here from MuJoCo's own kinematics, independently of the C++
        forward kinematics, so agreement validates the whole geometry chain.
        """
        if self.report_period_s <= 0.0:
            return
        now = float(data.time)
        if now - self._last_report < self.report_period_s:
            return
        self._last_report = now

        parts = []
        for name, body_id in zip(self.WRIST_BODIES, self._wrist_ids, strict=False):
            if body_id < 0:
                continue
            p = pelvis_mat.T @ (np.array(data.xpos[body_id], dtype=float) - pelvis_pos)
            parts.append(f"{name}=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
        if parts:
            print("[SharedAutonomy][sim] wrists in pelvis frame: " + "  ".join(parts))
