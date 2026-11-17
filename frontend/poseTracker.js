import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

export const POSE_CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
];

export class PoseTracker {
  static async create() {
    const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    const landmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "/models/pose_landmarker_lite.task" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    return new PoseTracker(landmarker);
  }

  constructor(landmarker) {
    this.landmarker = landmarker;
  }

  detect(video, timestamp) {
    const result = this.landmarker.detectForVideo(video, timestamp);
    return {
      landmarks: result.landmarks?.[0] ?? null,
      worldLandmarks: result.worldLandmarks?.[0] ?? null,
    };
  }

  close() {
    this.landmarker.close();
  }
}

export function isLandmarkUsable(landmark, threshold = 0.45) {
  return Boolean(
    landmark &&
      Number.isFinite(landmark.x) &&
      Number.isFinite(landmark.y) &&
      Number.isFinite(landmark.z) &&
      (landmark.visibility ?? landmark.presence ?? 1) >= threshold,
  );
}

export function createSyntheticPose(timestamp) {
  const landmarks = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 1,
    presence: 1,
  }));
  const worldLandmarks = landmarks.map((landmark) => ({ ...landmark }));
  const set = (index, x, y, worldX, worldY, worldZ = 0, visibility = 1) => {
    landmarks[index] = { x, y, z: 0, visibility, presence: visibility };
    worldLandmarks[index] = {
      x: worldX,
      y: worldY,
      z: worldZ,
      visibility,
      presence: visibility,
    };
  };

  const setNeutral = () => {
    set(0, 0.5, 0.18, 0, 1.3, -0.08);
    set(7, 0.53, 0.18, 0.08, 1.3);
    set(8, 0.47, 0.18, -0.08, 1.3);
    set(11, 0.59, 0.33, 0.22, 0.32);
    set(12, 0.41, 0.33, -0.22, 0.32);
    set(13, 0.6, 0.45, 0.24, 0.05);
    set(14, 0.4, 0.45, -0.24, 0.05);
    set(15, 0.6, 0.56, 0.25, -0.2);
    set(16, 0.4, 0.56, -0.25, -0.2);
    set(23, 0.55, 0.57, 0.12, 0);
    set(24, 0.45, 0.57, -0.12, 0);
    set(25, 0.55, 0.74, 0.12, -0.55);
    set(26, 0.45, 0.74, -0.12, -0.55);
    set(27, 0.55, 0.92, 0.12, -1.05);
    set(28, 0.45, 0.92, -0.12, -1.05);
  };
  setNeutral();

  const elapsed = (timestamp / 1000) % 34;
  const phase = Math.floor(elapsed / 2);
  const phaseTime = elapsed % 2;
  const phases = [
    "neutral",
    "left arm raised",
    "right arm raised",
    "both arms raised",
    "left elbow bent",
    "right elbow bent",
    "head tilted left",
    "head tilted right",
    "head turned left",
    "head turned right",
    "torso turned left",
    "torso turned right",
    "leaning forward",
    "leaning backward",
    "knees bent",
    "partial occlusion",
    phaseTime < 1 ? "tracking lost" : "reacquired",
  ];

  if (phase === 1) {
    set(13, 0.64, 0.24, 0.34, 0.55, 0.02);
    set(15, 0.7, 0.12, 0.4, 0.88, 0.04);
  } else if (phase === 2) {
    set(14, 0.36, 0.24, -0.34, 0.55, 0.02);
    set(16, 0.3, 0.12, -0.4, 0.88, 0.04);
  } else if (phase === 3) {
    set(13, 0.64, 0.24, 0.34, 0.55, 0.02);
    set(15, 0.7, 0.12, 0.4, 0.88, 0.04);
    set(14, 0.36, 0.24, -0.34, 0.55, 0.02);
    set(16, 0.3, 0.12, -0.4, 0.88, 0.04);
  } else if (phase === 4) {
    set(13, 0.55, 0.35, 0.08, 0.28);
    set(15, 0.54, 0.24, 0.04, 0.62);
  } else if (phase === 5) {
    set(14, 0.45, 0.35, -0.08, 0.28);
    set(16, 0.46, 0.24, -0.04, 0.62);
  } else if (phase === 6 || phase === 7) {
    const tilt = phase === 6 ? -1 : 1;
    set(7, 0.53, 0.18 + tilt * 0.04, 0.08, 1.3 + tilt * 0.08);
    set(8, 0.47, 0.18 - tilt * 0.04, -0.08, 1.3 - tilt * 0.08);
  } else if (phase === 8 || phase === 9) {
    const yaw = phase === 8 ? 0.65 : -0.65;
    const sin = Math.sin(yaw);
    const cos = Math.cos(yaw);
    set(0, 0.5 + sin * 0.08, 0.18, sin * 0.08, 1.3, -cos * 0.08);
    set(7, 0.5 + cos * 0.08, 0.18, cos * 0.08, 1.3, sin * 0.08);
    set(8, 0.5 - cos * 0.08, 0.18, -cos * 0.08, 1.3, -sin * 0.08);
  } else if (phase === 10 || phase === 11) {
    const yaw = phase === 10 ? 0.65 : -0.65;
    const sin = Math.sin(yaw);
    const cos = Math.cos(yaw);
    set(11, 0.5 + cos * 0.09, 0.33, cos * 0.22, 0.32, sin * 0.22);
    set(12, 0.5 - cos * 0.09, 0.33, -cos * 0.22, 0.32, -sin * 0.22);
    set(23, 0.5 + cos * 0.05, 0.57, cos * 0.12, 0, sin * 0.12);
    set(24, 0.5 - cos * 0.05, 0.57, -cos * 0.12, 0, -sin * 0.12);
  } else if (phase === 12 || phase === 13) {
    const lean = phase === 12 ? -0.18 : 0.18;
    set(11, 0.59, 0.33, 0.22, 0.32, lean);
    set(12, 0.41, 0.33, -0.22, 0.32, lean);
    set(0, 0.5, 0.18, 0, 1.3, lean * 1.7 - 0.08);
  } else if (phase === 14) {
    set(25, 0.56, 0.64, 0.2, -0.38, 0.1);
    set(26, 0.44, 0.65, -0.2, -0.38, -0.1);
    set(27, 0.58, 0.92, 0.3, -0.98, 0.2);
    set(28, 0.42, 0.92, -0.3, -0.98, -0.2);
  } else if (phase === 15) {
    set(13, 0.6, 0.45, 0.24, 0.05, 0, 0.1);
    set(15, 0.6, 0.56, 0.25, -0.2, 0, 0.1);
  } else if (phase === 16 && phaseTime < 1) {
    return { landmarks: null, worldLandmarks: null, phase: phases[phase] };
  }

  return { landmarks, worldLandmarks, phase: phases[phase] };
}
