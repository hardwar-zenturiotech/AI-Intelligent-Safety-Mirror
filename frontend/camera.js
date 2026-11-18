export async function requestCameraStream() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "Camera access is unavailable. Open this page on localhost or use a secure connection.",
    );
  }

  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280, max: 1920 },
      height: { ideal: 720, max: 1080 },
      frameRate: { ideal: 30, max: 60 },
    },
  });
}

export function mapLandmarkToStage(landmark, video, stage, mirrored = true) {
  if (!video.videoWidth || !video.videoHeight || !stage.width || !stage.height) {
    return null;
  }

  const scale = Math.max(stage.width / video.videoWidth, stage.height / video.videoHeight);
  const cropX = (stage.width - video.videoWidth * scale) / 2;
  const cropY = (stage.height - video.videoHeight * scale) / 2;
  const sourceX = landmark.x * video.videoWidth * scale + cropX;

  return {
    x: mirrored ? stage.width - sourceX : sourceX,
    y: landmark.y * video.videoHeight * scale + cropY,
  };
}
