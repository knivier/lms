const LOG_PATH = "../pose_log.jsonl";

const LABEL_MAP = {
  right_elbow: "Right arm (elbow)",
  left_elbow: "Left arm (elbow)",
  right_knee: "Right leg (knee)",
  left_knee: "Left leg (knee)",
  right_shoulder: "Right shoulder",
  left_shoulder: "Left shoulder",
  right_hip: "Right hip",
  left_hip: "Left hip",
  right_ankle: "Right ankle angle",
  left_ankle: "Left ankle angle",
  right_ankle_roll: "Right ankle roll",
  left_ankle_roll: "Left ankle roll",
  right_ankle_yaw: "Right ankle yaw",
  left_ankle_yaw: "Left ankle yaw",
  torso: "Torso",
};

const DEFAULT_SELECTED = new Set(["left_knee", "right_knee"]);

let chart = null;
let time = [];
let samples = [];
let angleKeys = [];

fetch(LOG_PATH)
  .then(res => res.text())
  .then(text => {
    const lines = text.trim().split("\n").filter(Boolean);
    for (const line of lines) {
      const obj = JSON.parse(line);
      time.push(obj.timestamp_ms / 1000);
      samples.push(obj.angles || {});
    }
    angleKeys = discoverKeys(samples);
    buildControls(angleKeys);
    plotSelected();
  });

function discoverKeys(sampleList) {
  const keys = new Set();
  for (const s of sampleList) {
    Object.keys(s || {}).forEach(k => keys.add(k));
  }
  return Array.from(keys).sort();
}

function buildControls(keys) {
  const container = document.getElementById("angleControls");
  container.innerHTML = "";
  keys.forEach(key => {
    const label = LABEL_MAP[key] || key;
    const wrapper = document.createElement("label");
    wrapper.style.display = "block";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = key;
    checkbox.checked = DEFAULT_SELECTED.has(key);
    checkbox.addEventListener("change", plotSelected);
    wrapper.appendChild(checkbox);
    wrapper.appendChild(document.createTextNode(` ${label}`));
    container.appendChild(wrapper);
  });
}

function getSelectedKeys() {
  return Array.from(document.querySelectorAll("#angleControls input[type=checkbox]:checked"))
    .map(el => el.value);
}

function buildDatasets(selected) {
  const palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
  ];
  return selected.map((key, idx) => {
    const data = samples.map(s => {
      const v = s[key];
      return v === null || v === undefined ? null : v;
    });
    return {
      label: `${LABEL_MAP[key] || key}`,
      data,
      borderColor: palette[idx % palette.length],
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.15,
      spanGaps: true
    };
  });
}

function plotSelected() {
  const ctx = document.getElementById("angleChart").getContext("2d");
  const selected = getSelectedKeys();
  const datasets = buildDatasets(selected);

  if (chart) {
    chart.data.labels = time;
    chart.data.datasets = datasets;
    chart.update();
    return;
  }

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: time,
      datasets
    },
    options: {
      animation: false,
      scales: {
        x: {
          title: { display: true, text: "Time (s)" }
        },
        y: {
          title: { display: true, text: "Angle (°)" }
        }
      }
    }
  });
}
