const ANGLE_KEYS = ["left_elbow", "right_elbow"];

fetch("../pose_log.jsonl")
  .then(res => res.text())
  .then(text => {
    const lines = text.trim().split("\n");

    const time = [];
    const byKey = {};
    for (const k of ANGLE_KEYS) byKey[k] = [];

    for (const line of lines) {
      const obj = JSON.parse(line);
      time.push(obj.timestamp_ms / 1000);
      for (const k of ANGLE_KEYS) {
        const v = obj.angles?.[k];
        byKey[k].push(v != null ? v : null);
      }
    }
    plot(time, byKey);
  });

function plot(time, byKey) {
  const ctx = document.getElementById("angleChart").getContext("2d");

  const colors = { left_elbow: "rgb(78, 121, 167)", right_elbow: "rgb(242, 142, 43)" };

  new Chart(ctx, {
    type: "line",
    data: {
      labels: time,
      datasets: ANGLE_KEYS.map((key) => ({
        label: key.replace("_", " "),
        data: byKey[key],
        borderColor: colors[key] || "rgb(100,100,100)",
        backgroundColor: (colors[key] || "rgb(100,100,100)").replace("rgb", "rgba").replace(")", ", 0.1)"),
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15,
        spanGaps: true
      }))
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
