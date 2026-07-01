const ASSET_DIR = "/Users/dewanshuhaswani/Documents/srib1/documentation/assets";

const C = {
  ink: "#071018",
  ink2: "#0d1622",
  panel: "#111c2a",
  panel2: "#172436",
  paper: "#eef6f8",
  white: "#f6fbff",
  muted: "#a8b8c8",
  cyan: "#5ee7ff",
  green: "#86f6b0",
  amber: "#ffd166",
  red: "#ff7b7b",
  blue: "#8db5ff",
  line: "#2b3b4f",
};

function img(name) {
  return `${ASSET_DIR}/${name}`;
}

function bg(slide, ctx, fill = C.ink) {
  ctx.addShape(slide, { x: 0, y: 0, width: 1280, height: 720, fill });
}

function footer(slide, ctx, n) {
  ctx.addShape(slide, { x: 48, y: 676, width: 132, height: 3, fill: C.cyan });
  ctx.addText(slide, {
    text: `GripSense RGB | AX Transformation | SRIB | ${String(n).padStart(2, "0")}`,
    x: 194,
    y: 662,
    width: 760,
    height: 24,
    fontSize: 10.5,
    color: "#8395a8",
  });
}

function kicker(slide, ctx, text, y = 36, color = C.cyan) {
  ctx.addShape(slide, { x: 52, y: y + 8, width: 8, height: 8, geometry: "ellipse", fill: color });
  ctx.addText(slide, {
    text,
    x: 68,
    y,
    width: 540,
    height: 28,
    fontSize: 12,
    bold: true,
    color,
    valign: "mid",
  });
}

function title(slide, ctx, text, sub, color = C.white) {
  ctx.addText(slide, {
    text,
    x: 52,
    y: 66,
    width: 790,
    height: 110,
    fontSize: 34,
    bold: true,
    color,
    typeface: ctx.fonts.title,
  });
  if (sub) {
    ctx.addText(slide, {
      text: sub,
      x: 54,
      y: 160,
      width: 760,
      height: 46,
      fontSize: 15.5,
      color: color === C.white ? "#b9c8d8" : "#4e6176",
    });
  }
}

function panel(slide, ctx, x, y, width, height, opts = {}) {
  ctx.addShape(slide, {
    x,
    y,
    width,
    height,
    fill: opts.fill || C.panel,
    line: { fill: opts.line || C.line, width: opts.lineWidth ?? 1 },
  });
  if (opts.accent) {
    ctx.addShape(slide, { x, y, width, height: 4, fill: opts.accent });
  }
}

function label(slide, ctx, text, x, y, width, color = C.cyan) {
  ctx.addText(slide, {
    text,
    x,
    y,
    width,
    height: 20,
    fontSize: 10.5,
    bold: true,
    color,
    valign: "mid",
  });
}

function body(slide, ctx, text, x, y, width, height, opts = {}) {
  ctx.addText(slide, {
    text,
    x,
    y,
    width,
    height,
    fontSize: opts.size || 14,
    bold: opts.bold || false,
    color: opts.color || "#d8e4ee",
    insets: opts.insets || { left: 0, right: 0, top: 0, bottom: 0 },
    valign: opts.valign || "top",
    align: opts.align || "left",
  });
}

function metric(slide, ctx, x, y, value, labelText, accent = C.cyan, width = 210) {
  panel(slide, ctx, x, y, width, 112, { fill: "#0f1a27", line: "#2a3b51", accent });
  body(slide, ctx, value, x + 18, y + 20, width - 36, 46, { size: 20, bold: true, color: C.white });
  body(slide, ctx, labelText, x + 18, y + 68, width - 34, 32, { size: 11.2, color: "#a9b9c9" });
}

function bulletList(slide, ctx, items, x, y, width, opts = {}) {
  items.forEach((item, index) => {
    const yy = y + index * (opts.gap || 48);
    ctx.addShape(slide, {
      x,
      y: yy + 5,
      width: 8,
      height: 8,
      geometry: "ellipse",
      fill: opts.accent || C.green,
    });
    body(slide, ctx, item, x + 22, yy, width - 22, opts.itemHeight || 38, {
      size: opts.size || 14,
      color: opts.color || "#dce8f2",
    });
  });
}

async function iconPill(slide, ctx, icon, text, x, y, width, color = C.cyan) {
  panel(slide, ctx, x, y, width, 48, { fill: "#111c2a", line: "#2c4058" });
  await ctx.addLucideIcon(slide, { icon, x: x + 14, y: y + 13, width: 22, height: 22, color, strokeWidth: 2.2 });
  body(slide, ctx, text, x + 46, y + 12, width - 54, 22, { size: 12.5, bold: true, color: C.white, valign: "mid" });
}

function connector(slide, ctx, x1, y1, x2, y2, color = C.cyan) {
  const width = Math.max(2, Math.abs(x2 - x1));
  const height = Math.max(2, Math.abs(y2 - y1));
  ctx.addShape(slide, {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    width,
    height,
    geometry: "line",
    fill: "#00000000",
    line: { fill: color, width: 2 },
  });
}

function stepNode(slide, ctx, x, y, width, titleText, detail, accent = C.cyan) {
  panel(slide, ctx, x, y, width, 108, { fill: "#111c2a", line: "#293b52", accent });
  body(slide, ctx, titleText, x + 16, y + 18, width - 32, 24, { size: 15, bold: true, color: C.white });
  body(slide, ctx, detail, x + 16, y + 50, width - 32, 44, { size: 11.5, color: "#aebdcb" });
}

function table(slide, ctx, x, y, rows, widths, opts = {}) {
  const rowH = opts.rowH || 48;
  rows.forEach((row, r) => {
    let xx = x;
    row.forEach((cell, c) => {
      const fill = r === 0 ? "#142234" : c === 0 ? "#101c2b" : "#0d1724";
      ctx.addShape(slide, {
        x: xx,
        y: y + r * rowH,
        width: widths[c],
        height: rowH,
        fill,
        line: { fill: "#273a50", width: 1 },
      });
      body(slide, ctx, cell, xx + 12, y + r * rowH + 11, widths[c] - 22, rowH - 14, {
        size: r === 0 ? 11.5 : 11.2,
        bold: r === 0 || c === 0,
        color: r === 0 ? C.cyan : c === 0 ? C.white : "#bdccd9",
      });
      xx += widths[c];
    });
  });
}

function placeholder(slide, ctx, x, y, width, height, titleText, detail) {
  panel(slide, ctx, x, y, width, height, { fill: "#0a121d", line: "#34506a" });
  ctx.addShape(slide, { x: x + 24, y: y + 24, width: width - 48, height: height - 48, fill: "#00000000", line: { fill: "#4e6b88", width: 2, style: "dash" } });
  body(slide, ctx, titleText, x + 50, y + height / 2 - 34, width - 100, 34, { size: 22, bold: true, color: C.white, align: "center" });
  body(slide, ctx, detail, x + 76, y + height / 2 + 4, width - 152, 44, { size: 13, color: "#b7c6d4", align: "center" });
}

async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  await ctx.addImage(slide, { path: img("gripsense-ax-concept.png"), x: 0, y: 0, width: 1280, height: 720, fit: "cover", alt: "Conceptual RGB grip analysis visual" });
  ctx.addShape(slide, { x: 0, y: 0, width: 680, height: 720, fill: "#071018dd" });
  ctx.addShape(slide, { x: 680, y: 0, width: 320, height: 720, fill: "#07101855" });
  kicker(slide, ctx, "AX TRANSFORMATION PIPELINE", 72);
  body(slide, ctx, "GripSense RGB", 58, 124, 540, 66, { size: 44, bold: true, color: C.white });
  body(slide, ctx, "RGB-only trained-object grip estimation for faster object-interaction research", 62, 206, 540, 72, { size: 22, color: "#d7e7f4" });
  body(slide, ctx, "Inventor: Dewanshu Haswani | Samsung Research Institute Bangalore (SRIB)", 64, 326, 520, 34, { size: 14, color: "#aab9c9" });
  metric(slide, ctx, 64, 430, "Live + offline", "webcam and uploaded video analysis", C.cyan, 176);
  metric(slide, ctx, 258, 430, "Object-first", "target identity gates grip scoring", C.green, 176);
  metric(slide, ctx, 452, 430, "Browser-first", "local privacy-preserving prototype", C.amber, 176);
  footer(slide, ctx, 1);
  return slide;
}

async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "EXECUTIVE THESIS");
  title(slide, ctx, "GripSense RGB turns ordinary cameras into an object-interaction intelligence layer.", "A browser-first platform that connects object identity, hand pose, grip evidence, and temporal stability so R&D teams can evaluate how people hold, move, and manipulate real products.");
  const items = [
    ["What it is", "A working RGB webcam/offline-video app for trained-object detection and visual grip-stability estimation."],
    ["Why it matters", "It reduces the friction of collecting interaction evidence before teams invest in depth hardware, smart objects, gloves, or lab rigs."],
    ["AX value", "It packages AI + automation into a reusable toolchain for experimentation, dataset creation, validation, and product-interaction analysis."],
  ];
  items.forEach(([h, b], i) => {
    const x = 68 + i * 390;
    panel(slide, ctx, x, 282, 340, 190, { fill: "#101b29", line: "#2a4058", accent: [C.cyan, C.green, C.amber][i] });
    body(slide, ctx, h, x + 22, 306, 290, 24, { size: 18, bold: true, color: C.white });
    body(slide, ctx, b, x + 22, 356, 292, 82, { size: 14.5, color: "#c4d4e3" });
  });
  body(slide, ctx, "The platform is honest by design: the percentage is visual grip stability, not measured physical force.", 90, 544, 1060, 30, { size: 16, bold: true, color: C.green, align: "center" });
  footer(slide, ctx, 2);
  return slide;
}

async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#f0f6f8");
  kicker(slide, ctx, "WHY AX", 36, "#0f8da0");
  title(slide, ctx, "The same interaction question keeps appearing across teams.", "Before automating, augmenting, simulating, or optimizing a task, teams need repeatable evidence of which object is present, how it is held, and whether the interaction is stable.", C.ink);
  const rows = [
    ["XR", "Can a headset understand hand-object interaction without markers?"],
    ["Visual Intelligence", "Can RGB models distinguish object presence from empty-hand false positives?"],
    ["Robotics", "Can demos capture grasp affordances, slip risk, and manipulation intent?"],
    ["On-device AI", "Can the perception pipeline run locally with privacy and latency constraints?"],
  ];
  rows.forEach(([team, question], i) => {
    const y = 212 + i * 95;
    ctx.addShape(slide, { x: 90, y, width: 140, height: 62, fill: [C.cyan, C.green, C.amber, C.blue][i] });
    body(slide, ctx, team, 108, y + 17, 100, 24, { size: 18, bold: true, color: C.ink, align: "center" });
    panel(slide, ctx, 250, y, 870, 62, { fill: "#ffffff", line: "#d3e2ea" });
    body(slide, ctx, question, 278, y + 16, 820, 26, { size: 18, color: "#1f2d3a" });
  });
  footer(slide, ctx, 3);
  return slide;
}

async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "TOOL OVERVIEW");
  title(slide, ctx, "Product flow: train object -> enable target -> detect target -> estimate grip.", "The UI separates object lock quality, object identity match, visual grip stability, and the limitation that RGB cannot measure true force.");
  await ctx.addImage(slide, { path: img("gripsense-v4-live-app.png"), x: 62, y: 202, width: 720, height: 450, fit: "contain", alt: "GripSense V4 app screenshot" });
  const flow = [
    ["1", "Train object profile", "Capture/crop/mask multiple angles and save locally."],
    ["2", "Enable target object", "Choose exactly what the app should track."],
    ["3", "Detect target", "Require temporal match before turning detection green."],
    ["4", "Estimate grip", "Score visible contact, wrap, support, and motion stability."],
  ];
  flow.forEach(([n, h, d], i) => {
    const y = 220 + i * 94;
    ctx.addShape(slide, { x: 830, y, width: 42, height: 42, geometry: "ellipse", fill: [C.cyan, C.green, C.amber, C.blue][i] });
    body(slide, ctx, n, 842, y + 10, 18, 18, { size: 15, bold: true, color: C.ink, align: "center" });
    body(slide, ctx, h, 894, y + 2, 260, 22, { size: 17, bold: true, color: C.white });
    body(slide, ctx, d, 894, y + 30, 300, 42, { size: 12.5, color: "#aebdcb" });
  });
  footer(slide, ctx, 4);
  return slide;
}

async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#08131e");
  kicker(slide, ctx, "SYSTEM PIPELINE");
  title(slide, ctx, "From RGB frames to team-readable grip evidence.", "The architecture is deliberately modular: stronger descriptor providers or a local inference server can replace the browser descriptor without rewriting the UI or scoring layer.");
  const nodes = [
    ["Camera / video", "webcam stream or uploaded clip", C.cyan],
    ["Hand landmarks", "MediaPipe Tasks Vision", C.green],
    ["Object profile", "masked crops, descriptors, augmentation", C.amber],
    ["Temporal identity", "multi-frame match and anti-flicker state", C.blue],
    ["Grip scoring", "mode-specific visual stability evidence", C.green],
    ["Export / review", "timeline, diagnostics, CSV/JSON", C.cyan],
  ];
  nodes.forEach(([h, d, accent], i) => {
    const x = 60 + i * 196;
    stepNode(slide, ctx, x, 300, 160, h, d, accent);
    if (i < nodes.length - 1) connector(slide, ctx, x + 160, 354, x + 190, 354, "#526c82");
  });
  panel(slide, ctx, 86, 508, 1080, 80, { fill: "#0f1a28", line: "#2b4158", accent: C.green });
  body(slide, ctx, "V4 design principle", 112, 526, 180, 22, { size: 15, bold: true, color: C.green });
  body(slide, ctx, "Never call a grip strong just because the hand is closed. First verify that the enabled target object is actually present and stable across several frames.", 300, 524, 840, 36, { size: 15, color: "#d5e5f0" });
  footer(slide, ctx, 5);
  return slide;
}

async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#eef6f8");
  kicker(slide, ctx, "IDENTITY FIRST", 36, "#0b8fa4");
  title(slide, ctx, "V4 fixes the core failure mode: empty hand should not become a confident object grip.", "The model now treats object identity and object lock as separate gates before visual grip scoring is allowed to look strong.", C.ink);
  await ctx.addImage(slide, { path: img("gripsense-field-hand-crop.png"), x: 76, y: 214, width: 315, height: 366, fit: "cover", alt: "Cropped false positive field observation" });
  label(slide, ctx, "Observed learning case", 76, 594, 280, "#0b8fa4");
  const gates = [
    ["Object lock", "Is the region real, not skin-only/background?"],
    ["Identity match", "Does it match an enabled trained object?"],
    ["Temporal match", "Has it persisted for repeated frames?"],
    ["Grip evidence", "Only then estimate stability and guidance."],
  ];
  gates.forEach(([h, d], i) => {
    const x = 458 + (i % 2) * 360;
    const y = 230 + Math.floor(i / 2) * 160;
    panel(slide, ctx, x, y, 300, 112, { fill: "#ffffff", line: "#d3e2ea", accent: [C.cyan, C.green, C.amber, C.blue][i] });
    body(slide, ctx, h, x + 20, y + 22, 250, 22, { size: 18, bold: true, color: C.ink });
    body(slide, ctx, d, x + 20, y + 54, 252, 38, { size: 13, color: "#4d6174" });
  });
  footer(slide, ctx, 6);
  return slide;
}

async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "OBJECT PROFILE V4");
  title(slide, ctx, "Training is a guided capture workflow, not a blind browser session.", "Users can add camera captures or uploads, crop/mask the object, name it, and save a profile locally for later use.");
  const roles = [
    ["Front view", "canonical product face"],
    ["Side view", "shape/aspect robustness"],
    ["Rotated view", "pose variance"],
    ["Object in hand", "occlusion reality"],
    ["Object alone", "clean descriptor anchor"],
    ["Negative examples", "nearby backgrounds / similar non-targets"],
  ];
  roles.forEach(([h, d], i) => {
    const x = 66 + (i % 3) * 390;
    const y = 220 + Math.floor(i / 3) * 150;
    panel(slide, ctx, x, y, 340, 112, { fill: "#101b29", line: "#2d4058", accent: [C.cyan, C.green, C.amber, C.blue, C.green, C.red][i] });
    body(slide, ctx, h, x + 20, y + 22, 250, 24, { size: 18, bold: true, color: C.white });
    body(slide, ctx, d, x + 20, y + 56, 280, 24, { size: 13.5, color: "#b7c7d5" });
  });
  panel(slide, ctx, 150, 542, 980, 48, { fill: "#0e1a27", line: "#2b4056" });
  body(slide, ctx, "Training is not blocked; the UI grades it as weak, good, or robust profile so experimentation can continue while quality remains visible.", 180, 556, 920, 20, { size: 14, color: C.green, bold: true, align: "center" });
  footer(slide, ctx, 7);
  return slide;
}

async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#f0f6f8");
  kicker(slide, ctx, "DESCRIPTOR MODEL", 36, "#0c8fa3");
  title(slide, ctx, "The browser descriptor is stronger than color matching, and prepared for embeddings.", "V4 uses handcrafted multi-signal descriptors now, while the ObjectDescriptorProvider interface keeps the path open for ONNX, CLIP, DINO, or a local Python server.", C.ink);
  table(slide, ctx, 70, 220, [
    ["Signal", "Current V4 browser implementation", "Why it improves accuracy"],
    ["Color", "HSV + normalized RGB/chroma histograms", "More robust under lighting shifts than raw RGB only"],
    ["Edges", "Orientation histograms + local gradients", "Catches phone/bottle silhouettes and side edges"],
    ["Shape", "Aspect, coverage, contour/tightness features", "Rejects huge boxes and background-like regions"],
    ["Texture", "Small grayscale patch statistics", "Adds weak material cues for remotes, mugs, tools"],
    ["Augmentation", "Flip/brightness/contrast variants", "Improves matching across camera exposure changes"],
  ], [170, 430, 440], { rowH: 58 });
  footer(slide, ctx, 8);
  return slide;
}

async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "TEMPORAL IDENTITY");
  title(slide, ctx, "Target detection is a state machine, not a one-frame decision.", "This reduces flicker, wrong detections, and the common issue where a closed hand looks like an object.");
  const stages = [
    ["No target", "no enabled profile"],
    ["Warming up", "candidate match appears"],
    ["Detected", "match streak passes threshold"],
    ["Grip active", "hand-object relationship is stable"],
    ["Lost/uncertain", "match drops or lock quality collapses"],
  ];
  stages.forEach(([h, d], i) => {
    const x = 70 + i * 230;
    ctx.addShape(slide, { x, y: 288, width: 130, height: 130, geometry: "ellipse", fill: ["#1a2a3a", "#263522", "#153325", "#17384a", "#3a2224"][i], line: { fill: [C.muted, C.amber, C.green, C.cyan, C.red][i], width: 3 } });
    body(slide, ctx, h, x + 15, 322, 100, 24, { size: 15, bold: true, color: C.white, align: "center" });
    body(slide, ctx, d, x + 14, 354, 102, 44, { size: 10.5, color: "#b9c6d2", align: "center" });
    if (i < stages.length - 1) connector(slide, ctx, x + 132, 353, x + 228, 353, "#536a82");
  });
  const bars = [["Object match", 86, C.green], ["Motion agreement", 72, C.cyan], ["Slip risk", 18, C.red], ["Lock quality", 81, C.amber]];
  bars.forEach(([h, v, color], i) => {
    const y = 514 + i * 34;
    body(slide, ctx, h, 160, y - 2, 160, 20, { size: 12, color: "#aebdca" });
    ctx.addShape(slide, { x: 330, y, width: 620, height: 10, fill: "#243244" });
    ctx.addShape(slide, { x: 330, y, width: Math.round(620 * v / 100), height: 10, fill: color });
    body(slide, ctx, `${v}%`, 970, y - 8, 60, 22, { size: 12, bold: true, color: C.white });
  });
  footer(slide, ctx, 9);
  return slide;
}

async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#eef6f8");
  kicker(slide, ctx, "VISUAL GRIP STABILITY", 36, "#0b8fa4");
  title(slide, ctx, "Grip percentage is a weighted evidence model, not force measurement.", "The score estimates whether the object appears visually secure in RGB frames by combining mode-specific features.", C.ink);
  const centerX = 360;
  const centerY = 390;
  ctx.addShape(slide, { x: centerX - 105, y: centerY - 105, width: 210, height: 210, geometry: "ellipse", fill: "#ffffff", line: { fill: "#b8cbd6", width: 2 } });
  body(slide, ctx, "Grip\nstability", centerX - 70, centerY - 32, 140, 64, { size: 22, bold: true, color: C.ink, align: "center" });
  const ev = [
    ["Visible contact", 360, 170, C.green],
    ["Finger wrap", 650, 270, C.cyan],
    ["Palm support", 650, 510, C.amber],
    ["Thumb support", 360, 620, C.blue],
    ["Motion stability", 78, 510, C.green],
    ["Slip persistence", 78, 270, C.red],
  ];
  ev.forEach(([h, x, y, color]) => {
    connector(slide, ctx, centerX, centerY, x + 80, y + 20, "#94a8ba");
    panel(slide, ctx, x, y, 170, 58, { fill: "#ffffff", line: "#cbd9e2", accent: color });
    body(slide, ctx, h, x + 12, y + 17, 146, 20, { size: 13, bold: true, color: C.ink, align: "center" });
  });
  panel(slide, ctx, 860, 232, 290, 312, { fill: "#ffffff", line: "#cbd9e2", accent: C.green });
  body(slide, ctx, "Guidance states", 884, 258, 220, 22, { size: 18, bold: true, color: C.ink });
  bulletList(slide, ctx, [
    "Strong hold: stable identity + strong visual evidence",
    "Improve grip: object detected, but support/contact is partial",
    "Object uncertain: detection/lock is the weak part",
    "Slip risk: sustained relative motion, not a noisy frame",
  ], 886, 312, 250, { accent: "#0b8fa4", color: "#425568", size: 12.3, gap: 48 });
  footer(slide, ctx, 10);
  return slide;
}

async function slide11(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "MODE-SPECIFIC SCORING");
  title(slide, ctx, "Different objects need different grip logic.", "A phone-side grip, bottle power grip, mug handle grip, pinch grip, and hook grip should not be scored with the same formula.");
  table(slide, ctx, 62, 216, [
    ["Grip mode", "Primary evidence", "Failure to avoid"],
    ["Phone / remote side grip", "side-edge support, finger segments, thumb-side opposition, temporal identity", "low score because only 1-2 fingertips are visible"],
    ["Bottle / power grip", "wrap, palm containment, coupling, slip persistence", "false slip when the object and hand are idle"],
    ["Pinch grip", "thumb-index opposition, small-object center stability", "penalizing small objects for low enclosure"],
    ["Hook grip", "finger curl and object support over fingertips", "requiring visible thumb when thumb is naturally absent"],
    ["Open hand", "low closure, no target identity, low containment", "empty hand becoming strong grip"],
  ], [230, 460, 510], { rowH: 64 });
  footer(slide, ctx, 11);
  return slide;
}

async function slide12(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#09121d");
  kicker(slide, ctx, "OFFLINE ANALYSIS");
  title(slide, ctx, "Uploaded videos become transparent, reviewable grip-analysis timelines.", "Offline mode supports repeatable analysis when live demo timing is difficult or when teams need a reportable record.");
  placeholder(slide, ctx, 76, 218, 560, 330, "LIVE DEMO VIDEO PLACEHOLDER", "Insert screen recording of GripSense RGB V4 here");
  const lines = [
    ["Grip %", [20, 35, 58, 76, 68, 86, 82, 61, 74], C.green],
    ["Object match", [5, 18, 54, 72, 88, 90, 85, 70, 76], C.cyan],
    ["Slip risk", [2, 4, 5, 8, 22, 14, 7, 35, 12], C.red],
  ];
  panel(slide, ctx, 710, 230, 380, 244, { fill: "#101b29", line: "#2b4058", accent: C.cyan });
  body(slide, ctx, "Timeline outputs", 734, 254, 260, 24, { size: 18, bold: true, color: C.white });
  lines.forEach(([h, pts, color], i) => {
    const yBase = 340 + i * 42;
    body(slide, ctx, h, 736, yBase - 20, 120, 18, { size: 11.5, color });
    pts.forEach((v, idx) => {
      const x = 850 + idx * 22;
      const hgt = Math.max(4, v * 0.42);
      ctx.addShape(slide, { x, y: yBase - hgt, width: 12, height: hgt, fill: color });
    });
  });
  await iconPill(slide, ctx, "Download", "Export CSV / JSON report", 744, 508, 250, C.green);
  footer(slide, ctx, 12);
  return slide;
}

async function slide13(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#eef6f8");
  kicker(slide, ctx, "PRODUCT FEATURES", 36, "#0b8fa4");
  title(slide, ctx, "V4 is already shaped as a working internal platform.", "It is not only an algorithm demo; it includes the product surfaces needed for training, evaluation, debugging, and reporting.", C.ink);
  await ctx.addImage(slide, { path: img("gripsense-v4-mobile.png"), x: 82, y: 190, width: 235, height: 510, fit: "cover", alt: "GripSense mobile UI screenshot" });
  const features = [
    ["Object profiles", "list trained objects, enable/disable target tracking"],
    ["Guided training", "camera capture/upload, crop/mask, multi-angle roles"],
    ["Live diagnostics", "object lock, identity match, grip stability, motion state"],
    ["Offline review", "video timeline, weak segments, slip events, report export"],
    ["Explainability", "info buttons and debug evidence for every score"],
    ["Privacy-first", "browser-local v1 path with optional stronger local server"],
  ];
  features.forEach(([h, d], i) => {
    const x = 382 + (i % 2) * 360;
    const y = 204 + Math.floor(i / 2) * 132;
    panel(slide, ctx, x, y, 310, 94, { fill: "#ffffff", line: "#d0dfe7", accent: [C.cyan, C.green, C.amber, C.blue, C.green, C.cyan][i] });
    body(slide, ctx, h, x + 18, y + 18, 250, 22, { size: 16.5, bold: true, color: C.ink });
    body(slide, ctx, d, x + 18, y + 48, 260, 30, { size: 12.2, color: "#526578" });
  });
  footer(slide, ctx, 13);
  return slide;
}

async function slide14(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "TEAM IMPACT: XR");
  title(slide, ctx, "XR can use GripSense as a rapid hand-object interaction probe.", "Useful before headset integration, especially for markerless object affordance experiments and human-object interaction datasets.");
  await iconPill(slide, ctx, "Eye", "Object-aware hand overlays for AR interaction prototypes", 90, 230, 470, C.cyan);
  await iconPill(slide, ctx, "Hand", "Grip-state cues for controller-free object manipulation", 90, 302, 470, C.green);
  await iconPill(slide, ctx, "Video", "Offline replay for comparing gestures across users/objects", 90, 374, 470, C.amber);
  await iconPill(slide, ctx, "Target", "Target-object gating to reduce false contextual actions", 90, 446, 470, C.blue);
  panel(slide, ctx, 700, 224, 330, 290, { fill: "#101b29", line: "#2c4058", accent: C.cyan });
  body(slide, ctx, "Example XR experiments", 728, 252, 260, 24, { size: 18, bold: true, color: C.white });
  bulletList(slide, ctx, [
    "detect when a mug/tool is being held vs touched",
    "compare phone-side grip stability before gesture trigger",
    "collect annotated affordance data without markers",
    "test AR hint overlays around suggested grip points",
  ], 732, 304, 270, { gap: 46, size: 12.2, accent: C.cyan });
  footer(slide, ctx, 14);
  return slide;
}

async function slide15(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#f0f6f8");
  kicker(slide, ctx, "TEAM IMPACT: VISUAL INTELLIGENCE", 36, "#0b8fa4");
  title(slide, ctx, "Visual Intelligence gets a practical benchmark for RGB object-hand reasoning.", "The tool creates reusable examples of hard cases: occlusion, poor lighting, target confusion, empty-hand false positives, and object-in-motion tracking.", C.ink);
  const cards = [
    ["Dataset creation", "Generate structured samples with object labels, masks, grip states, motion states, and frame timelines."],
    ["Model evaluation", "Compare browser descriptors, ONNX embeddings, CLIP/DINO-style local servers, and segmentation/detection models."],
    ["Failure mining", "Collect false positives/negatives as replayable cases for model improvement."],
    ["Explainability", "Score breakdown makes model behavior inspectable by engineers and product owners."],
  ];
  cards.forEach(([h, d], i) => {
    const x = 90 + (i % 2) * 520;
    const y = 218 + Math.floor(i / 2) * 176;
    panel(slide, ctx, x, y, 440, 128, { fill: "#ffffff", line: "#d4e2ea", accent: [C.cyan, C.green, C.amber, C.blue][i] });
    body(slide, ctx, h, x + 24, y + 24, 300, 24, { size: 18, bold: true, color: C.ink });
    body(slide, ctx, d, x + 24, y + 58, 370, 42, { size: 13.2, color: "#526578" });
  });
  footer(slide, ctx, 15);
  return slide;
}

async function slide16(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "TEAM IMPACT: ROBOTICS");
  title(slide, ctx, "Robotics can turn human demonstrations into grip and manipulation evidence.", "GripSense can help robotics teams study how people naturally hold objects before translating them to robot grasps, fixtures, or task policies.");
  const laneY = [238, 356, 474];
  const lanes = [
    ["Human demo", "record object handling with webcam/offline video", C.cyan],
    ["Grip evidence", "extract stable/weak/slip intervals and suggested contact zones", C.green],
    ["Robot insight", "inform grasp candidates, imitation learning labels, and manipulation tests", C.amber],
  ];
  lanes.forEach(([h, d, color], i) => {
    panel(slide, ctx, 105, laneY[i], 940, 70, { fill: "#101b29", line: "#2d4058", accent: color });
    body(slide, ctx, h, 135, laneY[i] + 18, 210, 24, { size: 18, bold: true, color: C.white });
    body(slide, ctx, d, 390, laneY[i] + 20, 580, 22, { size: 14, color: "#c1d0dc" });
    if (i < 2) connector(slide, ctx, 575, laneY[i] + 72, 575, laneY[i + 1] - 2, color);
  });
  await ctx.addLucideIcon(slide, { icon: "Bot", x: 1060, y: 312, width: 96, height: 96, color: C.green, strokeWidth: 1.8 });
  footer(slide, ctx, 16);
  return slide;
}

async function slide17(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#eef6f8");
  kicker(slide, ctx, "TEAM IMPACT: ON-DEVICE AI", 36, "#0b8fa4");
  title(slide, ctx, "On-device teams get a privacy-first benchmark for deployable perception.", "The current implementation keeps inference local in the browser, while the V4 interfaces are prepared for edge embeddings and model-compression experiments.", C.ink);
  metric(slide, ctx, 90, 226, "RGB-only", "works before depth/sensors are available", "#0b8fa4", 230);
  metric(slide, ctx, 360, 226, "Local-first", "no backend required for V4 prototype", C.green, 230);
  metric(slide, ctx, 630, 226, "Embedding-ready", "ONNX/local server descriptor path", C.amber, 230);
  metric(slide, ctx, 900, 226, "Explainable", "diagnostic scores map to modules", C.blue, 230);
  panel(slide, ctx, 140, 412, 980, 100, { fill: "#ffffff", line: "#d4e2ea", accent: C.green });
  body(slide, ctx, "On-device research questions it can support", 170, 434, 360, 22, { size: 18, bold: true, color: C.ink });
  body(slide, ctx, "What descriptor size is enough? How many frames are needed for stable identity? What is the latency budget for grip guidance? Which failure cases require depth, IMU, or sensor fusion?", 170, 468, 890, 34, { size: 13.5, color: "#526578" });
  footer(slide, ctx, 17);
  return slide;
}

async function slide18(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "AX IMPACT");
  title(slide, ctx, "The platform creates reusable leverage beyond one demo.", "GripSense can become a shared experimentation asset for product interaction, automation, model validation, and internal R&D storytelling.");
  const benefits = [
    ["Faster experiments", "turn webcam tests into structured observations in minutes"],
    ["Lower equipment cost", "start with RGB before depth rigs or smart objects"],
    ["Better datasets", "save object profiles, masks, examples, timelines, exports"],
    ["More honest AI", "uncertainty states prevent misleading high scores"],
    ["Team reuse", "same pipeline supports XR, VI, Robotics, and On-device AI"],
    ["Investor/IP signal", "novel object-first visual grip estimation workflow"],
  ];
  benefits.forEach(([h, d], i) => {
    const x = 76 + (i % 3) * 385;
    const y = 224 + Math.floor(i / 3) * 170;
    panel(slide, ctx, x, y, 315, 116, { fill: "#101b29", line: "#2b4058", accent: [C.cyan, C.green, C.amber, C.blue, C.green, C.cyan][i] });
    body(slide, ctx, h, x + 18, y + 22, 260, 22, { size: 16.2, bold: true, color: C.white });
    body(slide, ctx, d, x + 18, y + 54, 260, 36, { size: 12.5, color: "#b9c8d5" });
  });
  footer(slide, ctx, 18);
  return slide;
}

async function slide19(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#f0f6f8");
  kicker(slide, ctx, "NOVELTY", 36, "#0b8fa4");
  title(slide, ctx, "The novelty is in the productized perception loop, not a single isolated model.", "GripSense combines target-object training, temporal identity, grip-mode scoring, and explainable diagnostics into one live/offline workflow.", C.ink);
  const radar = [
    ["Target object identity", 92, C.green],
    ["Temporal stability", 86, C.cyan],
    ["Grip-mode evidence", 78, C.amber],
    ["Offline reports", 70, C.blue],
    ["Sensor-free start", 95, C.green],
  ];
  radar.forEach(([h, v, color], i) => {
    const y = 222 + i * 66;
    body(slide, ctx, h, 100, y, 220, 22, { size: 14.5, bold: true, color: C.ink });
    ctx.addShape(slide, { x: 330, y: y + 5, width: 570, height: 12, fill: "#d3e2ea" });
    ctx.addShape(slide, { x: 330, y: y + 5, width: Math.round(570 * v / 100), height: 12, fill: color });
    body(slide, ctx, `${v}%`, 920, y - 2, 70, 20, { size: 12.5, bold: true, color: C.ink });
  });
  panel(slide, ctx, 980, 226, 170, 300, { fill: "#ffffff", line: "#d4e2ea", accent: "#0b8fa4" });
  body(slide, ctx, "Research direction", 1000, 250, 130, 22, { size: 14.5, bold: true, color: C.ink });
  body(slide, ctx, "Move from RGB visual stability to calibrated grip confidence by adding embeddings, depth, IMU, pressure sensors, or smart-object data when needed.", 1000, 292, 126, 130, { size: 12.2, color: "#526578" });
  footer(slide, ctx, 19);
  return slide;
}

async function slide20(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "LIMITATIONS AND ROADMAP");
  title(slide, ctx, "The system is useful because it is honest about what RGB can and cannot know.", "The current app estimates visual stability; it does not directly measure physical force. That boundary keeps research conclusions defensible.");
  const limits = [
    ["RGB limitation", "cannot measure real pressure, torque, or hidden contact force"],
    ["Occlusion", "hidden fingers/object surfaces reduce evidence quality"],
    ["Lighting", "glare and low light can affect descriptors and landmarks"],
    ["Object ambiguity", "similar products need stronger embeddings or negatives"],
  ];
  limits.forEach(([h, d], i) => {
    panel(slide, ctx, 74, 218 + i * 78, 420, 56, { fill: "#101b29", line: "#2d4058", accent: C.red });
    body(slide, ctx, h, 94, 232 + i * 78, 150, 20, { size: 13.5, bold: true, color: C.white });
    body(slide, ctx, d, 250, 232 + i * 78, 230, 22, { size: 11.7, color: "#b9c8d5" });
  });
  const road = [
    ["Near", "V4 object profiles, better motion stability, offline reports"],
    ["Next", "browser ONNX embeddings or local CLIP/DINO server"],
    ["Future", "depth, IMU, pressure sensors, gloves, smart objects"],
  ];
  road.forEach(([h, d], i) => {
    const y = 236 + i * 120;
    ctx.addShape(slide, { x: 676, y: y + 10, width: 28, height: 28, geometry: "ellipse", fill: [C.green, C.cyan, C.amber][i] });
    connector(slide, ctx, 690, y + 38, 690, y + 104, "#49647d");
    body(slide, ctx, h, 730, y, 90, 24, { size: 18, bold: true, color: C.white });
    body(slide, ctx, d, 730, y + 34, 360, 36, { size: 13.5, color: "#bfd0dc" });
  });
  footer(slide, ctx, 20);
  return slide;
}

async function slide21(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, "#eef6f8");
  kicker(slide, ctx, "ADOPTION PLAN", 36, "#0b8fa4");
  title(slide, ctx, "A practical 90-day AX transformation path.", "Use the prototype to create measurable internal value quickly, then harden the perception stack where teams show demand.", C.ink);
  const phases = [
    ["0-30 days", "Internal demo and team feedback", "XR/VI/Robotics/On-device teams test common objects and log failure cases."],
    ["31-60 days", "Dataset and model bake-off", "Compare browser descriptor with ONNX/local embedding server on shared scenarios."],
    ["61-90 days", "Pilot workflows", "Offline reports, trained object registry, team-specific dashboards, acceptance criteria."],
  ];
  phases.forEach(([phase, h, d], i) => {
    const x = 76 + i * 385;
    panel(slide, ctx, x, 230, 320, 220, { fill: "#ffffff", line: "#d1e0e8", accent: [C.cyan, C.green, C.amber][i] });
    body(slide, ctx, phase, x + 22, 256, 180, 22, { size: 15, bold: true, color: "#0b8fa4" });
    body(slide, ctx, h, x + 22, 304, 260, 28, { size: 19, bold: true, color: C.ink });
    body(slide, ctx, d, x + 22, 356, 260, 66, { size: 13.2, color: "#526578" });
  });
  panel(slide, ctx, 154, 514, 972, 54, { fill: "#ffffff", line: "#d1e0e8", accent: C.green });
  body(slide, ctx, "Success metrics: false-positive reduction, stable detection latency, object-profile robustness, offline report completion rate, and number of team reuse cases.", 182, 530, 910, 22, { size: 14, bold: true, color: C.ink, align: "center" });
  footer(slide, ctx, 21);
  return slide;
}

async function slide22(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  kicker(slide, ctx, "LIVE DEMO AND ASK");
  title(slide, ctx, "Demo flow: train target object, enable it, move it, then review live and offline grip evidence.", "This slide intentionally leaves a clean placeholder for the final screen-recorded demo video.");
  placeholder(slide, ctx, 78, 208, 560, 340, "INSERT LIVE DEMO VIDEO", "Recommended: 45-75 seconds showing object training, target detection, grip score, motion stability, and offline export.");
  panel(slide, ctx, 720, 218, 330, 292, { fill: "#101b29", line: "#2d4058", accent: C.green });
  body(slide, ctx, "Submission ask", 750, 246, 260, 24, { size: 19, bold: true, color: C.white });
  bulletList(slide, ctx, [
    "Approve AX transformation submission as an internal reusable R&D tool",
    "Run pilot with XR, Visual Intelligence, Robotics, and On-device AI teams",
    "Collect shared test objects and failure cases",
    "Evaluate embedding-based V4/V5 upgrade path",
  ], 752, 300, 280, { gap: 48, size: 12.6, accent: C.green });
  body(slide, ctx, "Inventor: Dewanshu Haswani", 748, 548, 300, 22, { size: 15, bold: true, color: C.cyan });
  footer(slide, ctx, 22);
  return slide;
}

const slides = {
  1: slide01,
  2: slide02,
  3: slide03,
  4: slide04,
  5: slide05,
  6: slide06,
  7: slide07,
  8: slide08,
  9: slide09,
  10: slide10,
  11: slide11,
  12: slide12,
  13: slide13,
  14: slide14,
  15: slide15,
  16: slide16,
  17: slide17,
  18: slide18,
  19: slide19,
  20: slide20,
  21: slide21,
  22: slide22,
};

export async function makeSlide(presentation, ctx, slideNumber) {
  const fn = slides[slideNumber];
  if (!fn) throw new Error(`No slide registered for ${slideNumber}`);
  return fn(presentation, ctx);
}
