const ANGLE_KEY = "right_elbow"; // change to right_elbow, left_knee, etc.

fetch("../pose_log.jsonl")
  .then(res => res.text())
  .then(text => {
    const lines = text.trim().split("\n");

    const time = [];
    const angle = [];

    for (const line of lines) {
      const obj = JSON.parse(line);
      
      const value = obj.angles?.[ANGLE_KEY];
      if (value !== null && value !== undefined) {
        time.push(obj.timestamp_ms / 1000); // seconds
        angle.push(value);
      }
    }

    plot(time, angle);
  });

function plot(time, angle) {
  const ctx = document.getElementById("angleChart").getContext("2d");

  new Chart(ctx, {
    type: "line",
    data: {
      labels: time,
      datasets: [{
        label: `${ANGLE_KEY} Angle Over Time`,
        data: angle,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15
      }]
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
