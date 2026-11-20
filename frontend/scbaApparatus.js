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
export function createSCBACylinderApparatus(options = {}) {
  const apparatus = new THREE.Group();
  apparatus.name = "SCBA_Cylinder_Apparatus";

  // Reusable Materials
  const compositeDarkMat = new THREE.MeshStandardMaterial({
    color: 0x16191c,
    roughness: 0.7,
    metalness: 0.15,
  });

  const tankColor = options.color !== undefined ? options.color : 0xf5b800;
  const tankYellowMat = new THREE.MeshStandardMaterial({
    color: tankColor,
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
export function createSCBAFaceMask(options = {}) {
  const mask = new THREE.Group();
  mask.name = "SCBA_Face_Mask";

  const sealColor = options.rimColor !== undefined ? options.rimColor : 0x181a1c;
  const sealMat = new THREE.MeshStandardMaterial({
    color: sealColor,
    roughness: 0.65,
    metalness: 0.1,
  });

  const visorColor = options.visorColor !== undefined ? options.visorColor : 0xa4dcff;
  const visorMat = new THREE.MeshStandardMaterial({
    color: visorColor,
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

/**
 * Creates an authentic 3D Industrial Safety Helmet (Hard Hat).
 * ANSI Z89.1 / EN 397 Type 1 Class E compliant.
 * Includes:
 *  - High-impact polycarbonate/HDPE shell in OSHA high-visibility yellow
 *  - Reinforced top crown impact deflector ridge
 *  - Contoured front visor brim for face & glare protection
 *  - Flared rear nape skirt for neck protection
 *  - 360-degree rim / rain gutter flange
 *  - Dual retroreflective silver micro-prismatic safety stripes
 *  - Interior 4-point shock-absorbing suspension headband and brow sweatpad
 *  - Adjustable nylon web chin strap with quick-release buckle
 *
 * Attaches directly to the humanoid avatar's head bone.
 */
export function createSafetyHelmet(options = {}) {
  const helmet = new THREE.Group();
  helmet.name = "PPE_Safety_Helmet";

  const shellColor = options.color !== undefined ? options.color : 0xfac000;
  const shellYellowMat = new THREE.MeshStandardMaterial({
    color: shellColor,
    roughness: 0.22,
    metalness: 0.12,
  });

  const ridgeColor = options.ridgeColor !== undefined ? options.ridgeColor : 0xeb9800;
  const crownRidgeMat = new THREE.MeshStandardMaterial({
    color: ridgeColor,
    roughness: 0.25,
    metalness: 0.15,
  });

  const reflectiveSilverMat = new THREE.MeshStandardMaterial({
    color: 0xf0f2f5,
    roughness: 0.15,
    metalness: 0.55,
  });

  const suspensionBandMat = new THREE.MeshStandardMaterial({
    color: 0x22262a,
    roughness: 0.85,
    metalness: 0.1,
  });

  const strapMat = new THREE.MeshStandardMaterial({
    color: 0x1b1e22,
    roughness: 0.9,
    metalness: 0.05,
  });

  // 1. Main Dome Shell (contoured ellipsoid with generous clearance over skull and mask)
  const domeRadius = 0.115;
  const domeGeo = new THREE.SphereGeometry(
    domeRadius,
    32,
    20,
    0,
    Math.PI * 2,
    0,
    Math.PI * 0.52,
  );
  const domeMesh = new THREE.Mesh(domeGeo, shellYellowMat);
  domeMesh.scale.set(1.0, 0.98, 1.14);
  domeMesh.position.set(0, 0.02, 0);
  helmet.add(domeMesh);

  // 2. Crown Impact Deflection Ridge (anterior-to-posterior spine)
  const ridgeMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.024, 0.018, 0.19),
    crownRidgeMat,
  );
  ridgeMesh.position.set(0, 0.125, 0.005);
  helmet.add(ridgeMesh);

  // Secondary side strengthening ribs
  const ribL = new THREE.Mesh(
    new THREE.BoxGeometry(0.010, 0.010, 0.14),
    crownRidgeMat,
  );
  ribL.position.set(0.044, 0.11, 0.005);
  ribL.rotation.z = -0.3;
  helmet.add(ribL);

  const ribR = new THREE.Mesh(
    new THREE.BoxGeometry(0.010, 0.010, 0.14),
    crownRidgeMat,
  );
  ribR.position.set(-0.044, 0.11, 0.005);
  ribR.rotation.z = 0.3;
  helmet.add(ribR);

  // 3. Front Visor Brim (protrudes forward over brow)
  const visorBrim = new THREE.Mesh(
    new THREE.CylinderGeometry(0.125, 0.138, 0.008, 28, 1, false, -Math.PI * 0.35, Math.PI * 0.7),
    shellYellowMat,
  );
  visorBrim.position.set(0, 0.024, 0.028);
  visorBrim.scale.set(1.0, 1.0, 1.25);
  helmet.add(visorBrim);

  // 4. Rear Nape Protector (flared downward at back of skull)
  const napeSkirt = new THREE.Mesh(
    new THREE.CylinderGeometry(0.132, 0.120, 0.018, 24, 1, false, Math.PI * 0.65, Math.PI * 0.7),
    shellYellowMat,
  );
  napeSkirt.position.set(0, 0.012, -0.025);
  napeSkirt.rotation.x = 0.15;
  helmet.add(napeSkirt);

  // 5. Full-Perimeter Gutter / Rim Flange
  const rimTorus = new THREE.Mesh(
    new THREE.TorusGeometry(0.120, 0.006, 10, 36),
    shellYellowMat,
  );
  rimTorus.rotation.x = Math.PI * 0.5;
  rimTorus.scale.set(1.0, 1.15, 1.0);
  rimTorus.position.set(0, 0.02, 0);
  helmet.add(rimTorus);

  // 6. Retroreflective Micro-Prismatic Safety Stripes (Sides & Back)
  const stripeL = new THREE.Mesh(
    new THREE.BoxGeometry(0.004, 0.020, 0.09),
    reflectiveSilverMat,
  );
  stripeL.position.set(0.115, 0.062, 0);
  stripeL.rotation.y = -Math.PI * 0.05;
  helmet.add(stripeL);

  const stripeR = new THREE.Mesh(
    new THREE.BoxGeometry(0.004, 0.020, 0.09),
    reflectiveSilverMat,
  );
  stripeR.position.set(-0.115, 0.062, 0);
  stripeR.rotation.y = Math.PI * 0.05;
  helmet.add(stripeR);

  const stripeRear = new THREE.Mesh(
    new THREE.BoxGeometry(0.09, 0.018, 0.004),
    reflectiveSilverMat,
  );
  stripeRear.position.set(0, 0.052, -0.125);
  helmet.add(stripeRear);

  // 7. Interior Suspension Headband & Brow Sweatband
  const headband = new THREE.Mesh(
    new THREE.CylinderGeometry(0.105, 0.105, 0.020, 24, 1, true),
    suspensionBandMat,
  );
  headband.position.set(0, 0.018, 0.005);
  headband.scale.set(1.0, 1.0, 1.12);
  helmet.add(headband);

  // 8. Chin Strap (Y-loops dropping down from sides)
  const strapL = new THREE.Mesh(
    new THREE.BoxGeometry(0.004, 0.13, 0.012),
    strapMat,
  );
  strapL.position.set(0.102, -0.045, 0.01);
  strapL.rotation.z = -0.12;
  helmet.add(strapL);

  const strapR = new THREE.Mesh(
    new THREE.BoxGeometry(0.004, 0.13, 0.012),
    strapMat,
  );
  strapR.position.set(-0.102, -0.045, 0.01);
  strapR.rotation.z = 0.12;
  helmet.add(strapR);

  // Chin cup / buckle
  const chinCup = new THREE.Mesh(
    new THREE.BoxGeometry(0.055, 0.016, 0.022),
    strapMat,
  );
  chinCup.position.set(0, -0.105, 0.028);
  helmet.add(chinCup);

  helmet.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.renderOrder = 2;
    }
  });

  return helmet;
}

/**
 * Creates an authentic 3D High-Visibility Safety Vest.
 * ANSI/ISEA 107 Class 2 / ISO 20471 compliant.
 * Includes:
 *  - Generous clearance wrapping over the torso and shoulders without clipping
 *  - Breathable fluorescent neon yellow-green body panels
 *  - Heavy-duty reinforced black edge binding / contrast piping
 *  - Center front zipper / hook-and-loop closure strip
 *  - Dual continuous 360-degree horizontal retroreflective silver stripes
 *  - Dual vertical shoulder retroreflective silver stripes (harness layout)
 *  - Transparent credential/ID badge pocket on left chest
 *  - Radio mic clip tab and pen holder on right chest
 *
 * Attaches directly to the humanoid avatar's chest/spine bone.
 */
export function createSafetyVest(options = {}) {
  const vest = new THREE.Group();
  vest.name = "PPE_Safety_Vest";

  const fabricColor = options.color !== undefined ? options.color : 0xbcee00;
  const neonFabricMat = new THREE.MeshStandardMaterial({
    color: fabricColor, // High-visibility fluorescent fabric
    roughness: 0.8,
    metalness: 0.04,
  });

  const reflectiveSilverMat = new THREE.MeshStandardMaterial({
    color: 0xf2f4f8, // Micro-prismatic retroreflective silver
    roughness: 0.18,
    metalness: 0.6,
  });

  const blackTrimMat = new THREE.MeshStandardMaterial({
    color: 0x181a1c, // Heavy black polyester edge binding
    roughness: 0.9,
    metalness: 0.05,
  });

  const transparentIdMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.1,
    metalness: 0.1,
    transparent: true,
    opacity: 0.55,
  });

  // Generous wrap dimensions fitting outside the occluder torso (width 0.40, depth 0.22, height 0.34)
  const vestW = 0.40;
  const vestHalfW = vestW * 0.5;
  const vestH = 0.34;
  const vestFrontZ = 0.112;
  const vestBackZ = -0.102;

  // 1. Torso Main Enclosure
  // Front Left Panel
  const frontLeft = new THREE.Mesh(
    new THREE.BoxGeometry(0.185, vestH, 0.014),
    neonFabricMat,
  );
  frontLeft.position.set(0.10, 0, vestFrontZ);
  vest.add(frontLeft);

  // Front Right Panel
  const frontRight = new THREE.Mesh(
    new THREE.BoxGeometry(0.185, vestH, 0.014),
    neonFabricMat,
  );
  frontRight.position.set(-0.10, 0, vestFrontZ);
  vest.add(frontRight);

  // Center Closure Seam / Zipper (black vertical strip)
  const zipper = new THREE.Mesh(
    new THREE.BoxGeometry(0.018, vestH + 0.005, 0.018),
    blackTrimMat,
  );
  zipper.position.set(0, 0, vestFrontZ + 0.002);
  vest.add(zipper);

  // Back Panel (full back coverage from shoulders to waist)
  const backPanel = new THREE.Mesh(
    new THREE.BoxGeometry(vestW, vestH, 0.014),
    neonFabricMat,
  );
  backPanel.position.set(0, 0, vestBackZ);
  vest.add(backPanel);

  // Left Side Flank Panel (bridges front and back under armpit)
  const flankDepth = vestFrontZ - vestBackZ;
  const leftFlank = new THREE.Mesh(
    new THREE.BoxGeometry(0.014, vestH * 0.72, flankDepth),
    neonFabricMat,
  );
  leftFlank.position.set(vestHalfW - 0.007, -vestH * 0.14, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(leftFlank);

  // Right Side Flank Panel
  const rightFlank = new THREE.Mesh(
    new THREE.BoxGeometry(0.014, vestH * 0.72, flankDepth),
    neonFabricMat,
  );
  rightFlank.position.set(-vestHalfW + 0.007, -vestH * 0.14, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(rightFlank);

  // Left Shoulder Yoke / Strap (bridges over shoulder)
  const shoulderL = new THREE.Mesh(
    new THREE.BoxGeometry(0.10, 0.014, flankDepth + 0.01),
    neonFabricMat,
  );
  shoulderL.position.set(0.125, vestH * 0.5 - 0.007, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(shoulderL);

  // Right Shoulder Yoke / Strap
  const shoulderR = new THREE.Mesh(
    new THREE.BoxGeometry(0.10, 0.014, flankDepth + 0.01),
    neonFabricMat,
  );
  shoulderR.position.set(-0.125, vestH * 0.5 - 0.007, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(shoulderR);

  // 2. Reflective Prismatic Stripes
  // Lower Horizontal Stripe (around waist)
  const lowerY = -vestH * 0.28;
  const lowerStripeFront = new THREE.Mesh(
    new THREE.BoxGeometry(vestW, 0.028, 0.018),
    reflectiveSilverMat,
  );
  lowerStripeFront.position.set(0, lowerY, vestFrontZ + 0.002);
  vest.add(lowerStripeFront);

  const lowerStripeBack = new THREE.Mesh(
    new THREE.BoxGeometry(vestW, 0.028, 0.018),
    reflectiveSilverMat,
  );
  lowerStripeBack.position.set(0, lowerY, vestBackZ - 0.002);
  vest.add(lowerStripeBack);

  const lowerStripeL = new THREE.Mesh(
    new THREE.BoxGeometry(0.018, 0.028, flankDepth + 0.004),
    reflectiveSilverMat,
  );
  lowerStripeL.position.set(vestHalfW, lowerY, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(lowerStripeL);

  const lowerStripeR = new THREE.Mesh(
    new THREE.BoxGeometry(0.018, 0.028, flankDepth + 0.004),
    reflectiveSilverMat,
  );
  lowerStripeR.position.set(-vestHalfW, lowerY, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(lowerStripeR);

  // Mid Horizontal Stripe (around mid-chest)
  const midY = vestH * 0.05;
  const midStripeFront = new THREE.Mesh(
    new THREE.BoxGeometry(vestW, 0.028, 0.018),
    reflectiveSilverMat,
  );
  midStripeFront.position.set(0, midY, vestFrontZ + 0.002);
  vest.add(midStripeFront);

  const midStripeBack = new THREE.Mesh(
    new THREE.BoxGeometry(vestW, 0.028, 0.018),
    reflectiveSilverMat,
  );
  midStripeBack.position.set(0, midY, vestBackZ - 0.002);
  vest.add(midStripeBack);

  const midStripeL = new THREE.Mesh(
    new THREE.BoxGeometry(0.018, 0.028, flankDepth + 0.004),
    reflectiveSilverMat,
  );
  midStripeL.position.set(vestHalfW, midY, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(midStripeL);

  const midStripeR = new THREE.Mesh(
    new THREE.BoxGeometry(0.018, 0.028, flankDepth + 0.004),
    reflectiveSilverMat,
  );
  midStripeR.position.set(-vestHalfW, midY, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(midStripeR);

  // Vertical Stripes Over Left & Right Shoulders
  const stripeX = 0.115;
  const vertH = vestH * 0.45;
  const vertY = vestH * 0.26;

  const vertLFront = new THREE.Mesh(
    new THREE.BoxGeometry(0.028, vertH, 0.018),
    reflectiveSilverMat,
  );
  vertLFront.position.set(stripeX, vertY, vestFrontZ + 0.002);
  vest.add(vertLFront);

  const vertLOver = new THREE.Mesh(
    new THREE.BoxGeometry(0.028, 0.018, flankDepth + 0.014),
    reflectiveSilverMat,
  );
  vertLOver.position.set(stripeX, vestH * 0.5 + 0.002, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(vertLOver);

  const vertLBack = new THREE.Mesh(
    new THREE.BoxGeometry(0.028, vertH, 0.018),
    reflectiveSilverMat,
  );
  vertLBack.position.set(stripeX, vertY, vestBackZ - 0.002);
  vest.add(vertLBack);

  const vertRFront = new THREE.Mesh(
    new THREE.BoxGeometry(0.028, vertH, 0.018),
    reflectiveSilverMat,
  );
  vertRFront.position.set(-stripeX, vertY, vestFrontZ + 0.002);
  vest.add(vertRFront);

  const vertROver = new THREE.Mesh(
    new THREE.BoxGeometry(0.028, 0.018, flankDepth + 0.014),
    reflectiveSilverMat,
  );
  vertROver.position.set(-stripeX, vestH * 0.5 + 0.002, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(vertROver);

  const vertRBack = new THREE.Mesh(
    new THREE.BoxGeometry(0.028, vertH, 0.018),
    reflectiveSilverMat,
  );
  vertRBack.position.set(-stripeX, vertY, vestBackZ - 0.002);
  vest.add(vertRBack);

  // 3. Black Contrast Edge Piping Trim
  // Bottom hem
  const bottomTrim = new THREE.Mesh(
    new THREE.BoxGeometry(vestW + 0.01, 0.010, flankDepth + 0.01),
    blackTrimMat,
  );
  bottomTrim.position.set(0, -vestH * 0.5, (vestFrontZ + vestBackZ) * 0.5);
  vest.add(bottomTrim);

  // Collar V-trim
  const collarL = new THREE.Mesh(
    new THREE.BoxGeometry(0.010, 0.10, 0.018),
    blackTrimMat,
  );
  collarL.position.set(0.055, vestH * 0.35, vestFrontZ + 0.003);
  collarL.rotation.z = -0.55;
  vest.add(collarL);

  const collarR = new THREE.Mesh(
    new THREE.BoxGeometry(0.010, 0.10, 0.018),
    blackTrimMat,
  );
  collarR.position.set(-0.055, vestH * 0.35, vestFrontZ + 0.003);
  collarR.rotation.z = 0.55;
  vest.add(collarR);

  // 4. ID Badge Pocket on Left Chest
  const idBadge = new THREE.Mesh(
    new THREE.BoxGeometry(0.065, 0.08, 0.006),
    transparentIdMat,
  );
  idBadge.position.set(0.065, vestH * 0.14, vestFrontZ + 0.010);
  vest.add(idBadge);

  // Radio clip / pen tab on Right Chest
  const radioTab = new THREE.Mesh(
    new THREE.BoxGeometry(0.040, 0.014, 0.010),
    blackTrimMat,
  );
  radioTab.position.set(-0.065, vestH * 0.18, vestFrontZ + 0.010);
  vest.add(radioTab);

  vest.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.renderOrder = 2;
    }
  });

  return vest;
}

/**
 * Creates an authentic 3D Industrial Safety Gauntlet Glove.
 * EN 388 / ANSI Cut & Abrasion Resistant Work Glove.
 * Proportioned generously to wrap completely around the humanoid hand bones.
 * Includes:
 *  - High-visibility safety neon orange fabric back with breathable knit
 *  - Textured heavy-duty black nitrile/polyurethane grip palm & finger pads
 *  - Molded thermoplastic rubber (TPR) knuckle impact guard shield
 *  - TPR protective impact spines along each finger and thumb
 *  - Extended protective gauntlet cuff with elastic wrist cinch
 *  - 360-degree retroreflective silver wrist safety band
 *
 * Attaches directly to the humanoid avatar's leftHand (wristL) and rightHand (wristR) bones.
 * @param {"left"|"right"} hand
 */
export function createProtectiveGloves(handOrOptions = "left", maybeOptions = {}) {
  const hand = typeof handOrOptions === "string" ? handOrOptions : handOrOptions?.hand || "left";
  const options = typeof handOrOptions === "object" ? handOrOptions : maybeOptions;
  const glove = new THREE.Group();
  glove.name = `PPE_Protective_Glove_${hand.toUpperCase()}`;
  const isLeft = hand === "left";
  const sign = isLeft ? 1 : -1;

  const gloveColor = options.color !== undefined ? options.color : 0xf26522;
  const orangeHiVisMat = new THREE.MeshStandardMaterial({
    color: gloveColor, // High-visibility safety fabric
    roughness: 0.55,
    metalness: 0.08,
  });

  const gripMat = new THREE.MeshStandardMaterial({
    color: 0x1f2326, // Textured black nitrile/leather grip pad
    roughness: 0.88,
    metalness: 0.15,
  });

  const tprArmorMat = new THREE.MeshStandardMaterial({
    color: 0x141618, // Molded TPR impact armor shield
    roughness: 0.35,
    metalness: 0.3,
  });

  const reflectiveSilverMat = new THREE.MeshStandardMaterial({
    color: 0xf0f2f5, // Retroreflective silver wrist band
    roughness: 0.18,
    metalness: 0.55,
  });

  const cuffKnitMat = new THREE.MeshStandardMaterial({
    color: 0x2b3035, // Elastic ribbed gauntlet cuff
    roughness: 0.9,
    metalness: 0.05,
  });

  // 1. Extended Protective Gauntlet Cuff (covers wrist and lower forearm)
  const cuffMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.034, 0.038, 0.050, 18),
    cuffKnitMat,
  );
  cuffMesh.position.set(0, -0.012, 0);
  glove.add(cuffMesh);

  // Reflective Silver Band around Gauntlet Cuff
  const cuffReflective = new THREE.Mesh(
    new THREE.CylinderGeometry(0.0345, 0.0365, 0.014, 18),
    reflectiveSilverMat,
  );
  cuffReflective.position.set(0, -0.012, 0);
  glove.add(cuffReflective);

  // 2. Main Hand Body (Palm & Dorsal Casing, sized generously to encase hand)
  const palmW = 0.056;
  const palmH = 0.052;
  const palmD = 0.024;

  // Back of hand (Hi-Vis orange)
  const backHand = new THREE.Mesh(
    new THREE.BoxGeometry(palmW, palmH, palmD * 0.55),
    orangeHiVisMat,
  );
  backHand.position.set(0, 0.032, palmD * 0.25);
  glove.add(backHand);

  // Palm side (Grip nitrile)
  const palm = new THREE.Mesh(
    new THREE.BoxGeometry(palmW, palmH, palmD * 0.55),
    gripMat,
  );
  palm.position.set(0, 0.032, -palmD * 0.25);
  glove.add(palm);

  // 3. Molded TPR Knuckle Impact Armor Shield (across dorsal metacarpals)
  const knuckleShield = new THREE.Mesh(
    new THREE.BoxGeometry(palmW + 0.004, 0.016, 0.012),
    tprArmorMat,
  );
  knuckleShield.position.set(0, 0.052, palmD * 0.45);
  glove.add(knuckleShield);

  // 4 individual knuckle impact bumps
  for (let i = 0; i < 4; i++) {
    const bump = new THREE.Mesh(
      new THREE.SphereGeometry(0.006, 8, 8),
      tprArmorMat,
    );
    bump.position.set((-0.021 + i * 0.014) * sign, 0.052, palmD * 0.55);
    glove.add(bump);
  }

  // 4. 4 Finger Sheaths (Index, Middle, Ring, Pinky)
  const fingerWidth = 0.0125;
  const fingerDepth = 0.018;
  const fingerLengths = [0.046, 0.052, 0.048, 0.038]; // Index, Middle, Ring, Little

  for (let i = 0; i < 4; i++) {
    const fX = (-0.021 + i * 0.014) * sign;
    const fLen = fingerLengths[i];
    const fGroup = new THREE.Group();
    fGroup.position.set(fX, 0.058, 0);

    // Back of finger (orange)
    const fBack = new THREE.Mesh(
      new THREE.BoxGeometry(fingerWidth, fLen, fingerDepth * 0.5),
      orangeHiVisMat,
    );
    fBack.position.set(0, fLen * 0.5, fingerDepth * 0.25);
    fGroup.add(fBack);

    // Front/Palm of finger (black grip)
    const fPalm = new THREE.Mesh(
      new THREE.BoxGeometry(fingerWidth, fLen, fingerDepth * 0.5),
      gripMat,
    );
    fPalm.position.set(0, fLen * 0.5, -fingerDepth * 0.25);
    fGroup.add(fPalm);

    // Rounded fingertip cap
    const fTip = new THREE.Mesh(
      new THREE.SphereGeometry(fingerWidth * 0.5, 8, 8),
      gripMat,
    );
    fTip.position.set(0, fLen, 0);
    fGroup.add(fTip);

    // TPR Finger Impact Ribs (segmented dorsal protection)
    const rib1 = new THREE.Mesh(
      new THREE.BoxGeometry(fingerWidth * 0.85, 0.008, 0.006),
      tprArmorMat,
    );
    rib1.position.set(0, fLen * 0.35, fingerDepth * 0.45);
    fGroup.add(rib1);

    const rib2 = new THREE.Mesh(
      new THREE.BoxGeometry(fingerWidth * 0.85, 0.008, 0.006),
      tprArmorMat,
    );
    rib2.position.set(0, fLen * 0.70, fingerDepth * 0.45);
    fGroup.add(rib2);

    glove.add(fGroup);
  }

  // 5. Thumb Sheath (angled outward on thenar side)
  const thumbGroup = new THREE.Group();
  const thumbX = sign > 0 ? -0.028 : 0.028;
  thumbGroup.position.set(thumbX, 0.028, 0.008);
  thumbGroup.rotation.z = sign > 0 ? 0.45 : -0.45;
  thumbGroup.rotation.y = sign > 0 ? 0.3 : -0.3;

  const thumbLen = 0.040;
  const thumbBack = new THREE.Mesh(
    new THREE.BoxGeometry(0.013, thumbLen, 0.012),
    orangeHiVisMat,
  );
  thumbBack.position.set(0, thumbLen * 0.5, 0.006);
  thumbGroup.add(thumbBack);

  const thumbPalm = new THREE.Mesh(
    new THREE.BoxGeometry(0.013, thumbLen, 0.012),
    gripMat,
  );
  thumbPalm.position.set(0, thumbLen * 0.5, -0.006);
  thumbGroup.add(thumbPalm);

  const thumbTip = new THREE.Mesh(
    new THREE.SphereGeometry(0.0065, 8, 8),
    gripMat,
  );
  thumbTip.position.set(0, thumbLen, 0);
  thumbGroup.add(thumbTip);

  const thumbRib = new THREE.Mesh(
    new THREE.BoxGeometry(0.011, 0.009, 0.006),
    tprArmorMat,
  );
  thumbRib.position.set(0, thumbLen * 0.5, 0.010);
  thumbGroup.add(thumbRib);

  glove.add(thumbGroup);

  glove.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.renderOrder = 2;
    }
  });

  return glove;
}

