import * as THREE from "three";

/**
 * Creates an authentic 3D SCBA (Self-Contained Breathing Apparatus) Cylinder Pack.
 * Includes:
 *  - Ergonomic composite backplate & mounting frame
 *  - High-pressure composite air cylinder tank (high-vis safety yellow with reflective bands)
 *  - Main cylinder valve handwheel and neck manifold
 *  - First-stage pressure reducer block and flexible braided air supply hose
 *  - Shoulder-mounted pressure gauge / PASS distress unit (with green 300-bar dial)
 *  - Padded ergonomic shoulder straps and waist harness belt with safety buckle
 *
 * Designed to attach directly to the humanoid avatar's chest/spine bone.
 */
export function createSCBACylinderApparatus() {
  const apparatus = new THREE.Group();
  apparatus.name = "SCBA_Cylinder_Apparatus";

  // Reusable Materials
  const compositeDarkMat = new THREE.MeshStandardMaterial({
    color: 0x16191c,
    roughness: 0.7,
    metalness: 0.15,
  });

  const tankYellowMat = new THREE.MeshStandardMaterial({
    color: 0xf5b800,
    roughness: 0.28,
    metalness: 0.12,
  });

  const carbonBandMat = new THREE.MeshStandardMaterial({
    color: 0x24282c,
    roughness: 0.5,
    metalness: 0.3,
  });

  const reflectiveStripeMat = new THREE.MeshStandardMaterial({
    color: 0xe8eaec,
    roughness: 0.2,
    metalness: 0.4,
  });

  const chromeMat = new THREE.MeshStandardMaterial({
    color: 0xcccccc,
    roughness: 0.18,
    metalness: 0.9,
  });

  const brassValveMat = new THREE.MeshStandardMaterial({
    color: 0x8e825a,
    roughness: 0.3,
    metalness: 0.8,
  });

  const hoseMat = new THREE.MeshStandardMaterial({
    color: 0x1e2022,
    roughness: 0.85,
    metalness: 0.05,
  });

  const strapMat = new THREE.MeshStandardMaterial({
    color: 0x22262a,
    roughness: 0.8,
    metalness: 0.1,
  });

  // 1. Backplate & Frame (contoured against human back)
  const backplateGroup = new THREE.Group();
  const backplateMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.24, 0.46, 0.022),
    compositeDarkMat,
  );
  backplateMesh.castShadow = true;
  backplateGroup.add(backplateMesh);

  // Lumbar pad at lower back
  const lumbarPad = new THREE.Mesh(
    new THREE.BoxGeometry(0.22, 0.08, 0.03),
    strapMat,
  );
  lumbarPad.position.set(0, -0.19, 0.01);
  backplateGroup.add(lumbarPad);

  // Top neck cradle
  const topCradle = new THREE.Mesh(
    new THREE.BoxGeometry(0.16, 0.05, 0.03),
    compositeDarkMat,
  );
  topCradle.position.set(0, 0.19, -0.015);
  backplateGroup.add(topCradle);

  apparatus.add(backplateGroup);

  // 2. High-Pressure Air Cylinder Tank
  const tankGroup = new THREE.Group();
  tankGroup.position.set(0, 0.02, -0.082); // Extends outward on back

  const tankRadius = 0.062;
  const cylinderHeight = 0.34;

  // Main cylindrical body
  const tankBody = new THREE.Mesh(
    new THREE.CylinderGeometry(tankRadius, tankRadius, cylinderHeight, 28),
    tankYellowMat,
  );
  tankBody.castShadow = true;
  tankGroup.add(tankBody);

  // Top hemispherical dome
  const topDome = new THREE.Mesh(
    new THREE.SphereGeometry(tankRadius, 28, 16, 0, Math.PI * 2, 0, Math.PI / 2),
    tankYellowMat,
  );
  topDome.position.y = cylinderHeight / 2;
  tankGroup.add(topDome);

  // Bottom hemispherical dome
  const bottomDome = new THREE.Mesh(
    new THREE.SphereGeometry(tankRadius, 28, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2),
    tankYellowMat,
  );
  bottomDome.position.y = -cylinderHeight / 2;
  tankGroup.add(bottomDome);

  // Decorative carbon protective bands & reflective stripes
  const bandTop = new THREE.Mesh(
    new THREE.CylinderGeometry(tankRadius + 0.002, tankRadius + 0.002, 0.045, 28),
    carbonBandMat,
  );
  bandTop.position.y = 0.08;
  tankGroup.add(bandTop);

  const stripeTop = new THREE.Mesh(
    new THREE.CylinderGeometry(tankRadius + 0.003, tankRadius + 0.003, 0.015, 28),
    reflectiveStripeMat,
  );
  stripeTop.position.y = 0.08;
  tankGroup.add(stripeTop);

  const bandBottom = new THREE.Mesh(
    new THREE.CylinderGeometry(tankRadius + 0.002, tankRadius + 0.002, 0.045, 28),
    carbonBandMat,
  );
  bandBottom.position.y = -0.08;
  tankGroup.add(bandBottom);

  const stripeBottom = new THREE.Mesh(
    new THREE.CylinderGeometry(tankRadius + 0.003, tankRadius + 0.003, 0.015, 28),
    reflectiveStripeMat,
  );
  stripeBottom.position.y = -0.08;
  tankGroup.add(stripeBottom);

  // Retention strap across the backplate holding the cylinder
  const retentionStrap = new THREE.Mesh(
    new THREE.CylinderGeometry(tankRadius + 0.006, tankRadius + 0.006, 0.035, 28, 1, true, 0, Math.PI * 1.5),
    carbonBandMat,
  );
  retentionStrap.position.set(0, 0, 0);
  retentionStrap.rotation.y = Math.PI * 0.75;
  tankGroup.add(retentionStrap);

  const camBuckle = new THREE.Mesh(
    new THREE.BoxGeometry(0.03, 0.035, 0.015),
    chromeMat,
  );
  camBuckle.position.set(tankRadius + 0.008, 0, 0);
  tankGroup.add(camBuckle);

  // 3. Cylinder Valve & Handwheel Assembly (at top neck)
  const valveNeck = new THREE.Mesh(
    new THREE.CylinderGeometry(0.022, 0.025, 0.04, 16),
    chromeMat,
  );
  valveNeck.position.y = cylinderHeight / 2 + tankRadius + 0.015;
  tankGroup.add(valveNeck);

  const valveBody = new THREE.Mesh(
    new THREE.BoxGeometry(0.042, 0.042, 0.035),
    brassValveMat,
  );
  valveBody.position.y = cylinderHeight / 2 + tankRadius + 0.04;
  tankGroup.add(valveBody);

  const handwheel = new THREE.Mesh(
    new THREE.CylinderGeometry(0.026, 0.026, 0.022, 18),
    compositeDarkMat,
  );
  handwheel.position.set(0.032, cylinderHeight / 2 + tankRadius + 0.04, 0);
  handwheel.rotation.z = Math.PI / 2;
  tankGroup.add(handwheel);

  // 4. Pressure Reducer & Pneumatics Manifold (at bottom)
  const reducerBlock = new THREE.Mesh(
    new THREE.BoxGeometry(0.065, 0.05, 0.045),
    compositeDarkMat,
  );
  reducerBlock.position.set(0, -cylinderHeight / 2 - tankRadius - 0.02, 0.02);
  tankGroup.add(reducerBlock);

  apparatus.add(tankGroup);

  // 5. Shoulder Harness Straps (wrap forward over shoulders towards chest)
  const leftStrap = new THREE.Mesh(
    new THREE.CylinderGeometry(0.022, 0.026, 0.28, 12),
    strapMat,
  );
  leftStrap.position.set(0.10, 0.12, 0.06);
  leftStrap.rotation.x = Math.PI * 0.28;
  apparatus.add(leftStrap);

  const rightStrap = new THREE.Mesh(
    new THREE.CylinderGeometry(0.022, 0.026, 0.28, 12),
    strapMat,
  );
  rightStrap.position.set(-0.10, 0.12, 0.06);
  rightStrap.rotation.x = Math.PI * 0.28;
  apparatus.add(rightStrap);

  // Waist belt wrapping around hips
  const waistBeltLeft = new THREE.Mesh(
    new THREE.BoxGeometry(0.018, 0.065, 0.20),
    strapMat,
  );
  waistBeltLeft.position.set(0.12, -0.19, 0.08);
  waistBeltLeft.rotation.y = -Math.PI * 0.15;
  apparatus.add(waistBeltLeft);

  const waistBeltRight = new THREE.Mesh(
    new THREE.BoxGeometry(0.018, 0.065, 0.20),
    strapMat,
  );
  waistBeltRight.position.set(-0.12, -0.19, 0.08);
  waistBeltRight.rotation.y = Math.PI * 0.15;
  apparatus.add(waistBeltRight);

  // Central quick-release waist buckle
  const waistBuckle = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, 0.05, 0.02),
    chromeMat,
  );
  waistBuckle.position.set(0, -0.19, 0.17);
  apparatus.add(waistBuckle);

  // 6. Flexible High-Pressure Air Supply Hose & Pressure Gauge
  // Braided hose looping from bottom reducer over the left shoulder
  const hoseCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.02, -0.22, -0.06),
    new THREE.Vector3(0.12, -0.10, -0.04),
    new THREE.Vector3(0.13, 0.10, 0.02),
    new THREE.Vector3(0.10, 0.20, 0.10),
    new THREE.Vector3(0.07, 0.14, 0.15),
  ]);
  const hoseGeo = new THREE.TubeGeometry(hoseCurve, 20, 0.009, 8, false);
  const hoseMesh = new THREE.Mesh(hoseGeo, hoseMat);
  apparatus.add(hoseMesh);

  // Pressure Gauge & Integrated PASS Unit (on chest strap)
  const gaugeHousing = new THREE.Mesh(
    new THREE.CylinderGeometry(0.026, 0.026, 0.018, 20),
    compositeDarkMat,
  );
  gaugeHousing.position.set(0.07, 0.12, 0.16);
  gaugeHousing.rotation.x = Math.PI * 0.45;
  apparatus.add(gaugeHousing);

  // Luminescent green gauge face (showing 300-bar full air pressure)
  const gaugeFace = new THREE.Mesh(
    new THREE.CircleGeometry(0.022, 20),
    new THREE.MeshBasicMaterial({ color: 0x44dd66 }),
  );
  gaugeFace.position.set(0.07, 0.12, 0.17);
  gaugeFace.rotation.x = -Math.PI * 0.05;
  apparatus.add(gaugeFace);

  // Gauge needle (at 300 bar / 12 o'clock)
  const needle = new THREE.Mesh(
    new THREE.BoxGeometry(0.003, 0.016, 0.002),
    new THREE.MeshBasicMaterial({ color: 0x111111 }),
  );
  needle.position.set(0.07, 0.125, 0.172);
  apparatus.add(needle);

  return apparatus;
}

/**
 * Creates an authentic full-face industrial SCBA respirator mask.
 * Includes:
 *  - Panoramic clear polycarbonate visor lens
 *  - Ergonomic silicone face seal
 *  - Demand valve air regulator at mouth with purge button
 *  - Speech diaphragm / exhalation valve at chin
 *  - 5-point head harness strapping
 *
 * Designed to attach directly to the humanoid avatar's head bone.
 */
export function createSCBAFaceMask() {
  const mask = new THREE.Group();
  mask.name = "SCBA_Face_Mask";

  const sealMat = new THREE.MeshStandardMaterial({
    color: 0x181a1c,
    roughness: 0.65,
    metalness: 0.1,
  });

  const visorMat = new THREE.MeshStandardMaterial({
    color: 0xa4dcff,
    roughness: 0.1,
    metalness: 0.15,
    transparent: true,
    opacity: 0.58,
  });

  const regulatorMat = new THREE.MeshStandardMaterial({
    color: 0x282c30,
    roughness: 0.4,
    metalness: 0.3,
  });

  const purgeButtonMat = new THREE.MeshStandardMaterial({
    color: 0xe03b24, // Emergency purge button (red)
    roughness: 0.3,
    metalness: 0.1,
  });

  const chromeMat = new THREE.MeshStandardMaterial({
    color: 0xd0d4d8,
    roughness: 0.2,
    metalness: 0.85,
  });

  // 1. Panoramic Visor Lens (covers eyes, nose, upper face)
  const visor = new THREE.Mesh(
    new THREE.SphereGeometry(0.095, 24, 18, 0, Math.PI, 0, Math.PI * 0.65),
    visorMat,
  );
  visor.position.set(0, 0.02, 0.04);
  visor.rotation.x = Math.PI * 0.1;
  mask.add(visor);

  // 2. Silicone Outer Face Seal Frame
  const frameRim = new THREE.Mesh(
    new THREE.TorusGeometry(0.095, 0.014, 12, 28),
    sealMat,
  );
  frameRim.position.set(0, 0.02, 0.038);
  frameRim.scale.set(1.0, 1.15, 0.7);
  mask.add(frameRim);

  // 3. Lower Chin Cup & Speech Diaphragm
  const chinCup = new THREE.Mesh(
    new THREE.CylinderGeometry(0.05, 0.035, 0.055, 18),
    sealMat,
  );
  chinCup.position.set(0, -0.065, 0.075);
  chinCup.rotation.x = Math.PI * 0.25;
  mask.add(chinCup);

  const speechDiaphragm = new THREE.Mesh(
    new THREE.CylinderGeometry(0.022, 0.022, 0.01, 16),
    chromeMat,
  );
  speechDiaphragm.position.set(0, -0.065, 0.095);
  speechDiaphragm.rotation.x = Math.PI * 0.5;
  mask.add(speechDiaphragm);

  // 4. Demand Valve Regulator (at mouth / nose)
  const regulator = new THREE.Mesh(
    new THREE.CylinderGeometry(0.032, 0.032, 0.045, 20),
    regulatorMat,
  );
  regulator.position.set(0, -0.02, 0.115);
  regulator.rotation.x = Math.PI * 0.5;
  mask.add(regulator);

  // Emergency Purge Button on front of regulator
  const purgeButton = new THREE.Mesh(
    new THREE.CylinderGeometry(0.018, 0.018, 0.008, 18),
    purgeButtonMat,
  );
  purgeButton.position.set(0, -0.02, 0.14);
  purgeButton.rotation.x = Math.PI * 0.5;
  mask.add(purgeButton);

  // Air supply quick-connect fitting on side of regulator
  const fitting = new THREE.Mesh(
    new THREE.CylinderGeometry(0.01, 0.01, 0.025, 12),
    chromeMat,
  );
  fitting.position.set(0.038, -0.02, 0.115);
  fitting.rotation.z = Math.PI * 0.5;
  mask.add(fitting);

  // 5. 5-Point Head Harness Straps (wrap back over skull)
  const topStrap = new THREE.Mesh(
    new THREE.BoxGeometry(0.02, 0.006, 0.18),
    sealMat,
  );
  topStrap.position.set(0, 0.11, -0.02);
  topStrap.rotation.x = -Math.PI * 0.15;
  mask.add(topStrap);

  const leftStrap = new THREE.Mesh(
    new THREE.BoxGeometry(0.014, 0.006, 0.16),
    sealMat,
  );
  leftStrap.position.set(0.08, 0.02, -0.02);
  leftStrap.rotation.y = -Math.PI * 0.2;
  mask.add(leftStrap);

  const rightStrap = new THREE.Mesh(
    new THREE.BoxGeometry(0.014, 0.006, 0.16),
    sealMat,
  );
  rightStrap.position.set(-0.08, 0.02, -0.02);
  rightStrap.rotation.y = Math.PI * 0.2;
  mask.add(rightStrap);

  return mask;
}
