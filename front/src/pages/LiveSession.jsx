import { Box, Button, Chip, Stack, Typography, Fade, Grow, Slide, Paper, Grid } from "@mui/material";
import { useState, useRef, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import TimerIcon from "@mui/icons-material/Timer";
import AddIcon from "@mui/icons-material/Add";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import FeedbackIcon from "@mui/icons-material/Feedback";
// Tauri APIs are loaded dynamically when starting the feed so a missing/failing API never blanks the page.

// Metrics config per workout (title + exercise-specific metric placeholders)
const WORKOUT_METRICS = {
  squat: { title: "Squat", extra: "Depth: — · Knees: —", gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" },
  pushup: { title: "Push-up", extra: "Chest touch: — · Lockout: —", gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)" },
  bicep_curl: { title: "Bicep curl", extra: "ROM: — · Elbow drift: —", gradient: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)" },
};

/**
 * Backend stub: commit current set and get computed metrics.
 * Later: POST to backend with { workoutId, setStartTime, setEndTime, repTimes, formIssues }; return backend response.
 */
function commitSetStub(payload) {
  const { setStartTime, setEndTime, repTimes } = payload;
  const n = repTimes.length;
  const reps = n;
  const lastRepSec =
    n >= 2 ? ((repTimes[n - 1] - repTimes[n - 2]) / 1000).toFixed(1) : null;
  const avgSec =
    n >= 2
      ? ((repTimes[n - 1] - repTimes[0]) / 1000 / (n - 1)).toFixed(1)
      : null;
  return { reps, lastRepSec, avgSec };
}

export default function LiveSession() {
  const location = useLocation();
  const navigate = useNavigate();
  const workoutId = location.state?.workoutId ?? null;

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [cvStreamError, setCvStreamError] = useState(false);
  // Native feed (Tauri): cv.py frames from IPC. Fallback: stream URL when running in browser.
  const [cvFrame, setCvFrame] = useState(null);
  const unlistenCvFrameRef = useRef(null);
  const latestCvFrameRef = useRef(null);
  const cvFrameRafRef = useRef(null);

  const CV_STREAM_URL = "http://127.0.0.1:8765/cv-stream";
  const isTauri = typeof window !== "undefined" && (window.__TAURI_INTERNALS__ != null || window.__TAURI__ != null);

  // Current set: rep timestamps (CV or + Rep). Set boundary = first rep until "Stop current set".
  const [repTimes, setRepTimes] = useState([]);
  const setStartRef = useRef(null);

  // Last completed set metrics (displayed until next "Stop current set")
  const [lastSetMetrics, setLastSetMetrics] = useState(null);
  // All completed sets in this workout (for summary when user hits "End Workout")
  const [completedSets, setCompletedSets] = useState([]);

  // Form issues during current set (CV pushes here; sent to backend on "Stop current set"). Shape: { severity, message, joint }[]
  const [formIssuesDuringSet, setFormIssuesDuringSet] = useState([]);

  // Rest timing (starts when user hits "Stop current set")
  const [restTimestamps, setRestTimestamps] = useState([]);
  const [lastRestAt, setLastRestAt] = useState(null);
  const [timeSinceRest, setTimeSinceRest] = useState(0);
  const sessionStartRef = useRef(null);

  // After "End Workout": show summary; "Done" dismisses and resets for next workout
  const [showSummary, setShowSummary] = useState(false);

  // True when resting between sets (after "Stop current set", until "Start next set"); drives "Start next set" visibility and Rest label
  const [betweenSetsMode, setBetweenSetsMode] = useState(false);

  // Reset set state when user selects a different workout
  useEffect(() => {
    setRepTimes([]);
    setLastSetMetrics(null);
    setCompletedSets([]);
    setFormIssuesDuringSet([]);
    setShowSummary(false);
    setBetweenSetsMode(false);
    setStartRef.current = null;
  }, [workoutId]);

  // Persist selected workout to workout_id.json (Tauri only); updates whenever workoutId changes.
  useEffect(() => {
    if (typeof window === "undefined" || (window.__TAURI_INTERNALS__ == null && window.__TAURI__ == null)) return;
    import("@tauri-apps/api/core")
      .then((m) => m.invoke("write_workout_id", { workoutId: workoutId ?? "" }))
      .catch(() => {});
  }, [workoutId]);

  // Time since last rest (updates every second when running)
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      if (lastRestAt != null) {
        setTimeSinceRest(Math.floor((Date.now() - lastRestAt) / 1000));
      } else if (sessionStartRef.current != null) {
        setTimeSinceRest(Math.floor((Date.now() - sessionStartRef.current) / 1000));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [isRunning, lastRestAt]);

  // Native feed (Tauri): start cv.py pipeline and listen for frames; stop and unlisten on end.
  // Store latest frame in a ref and paint at rAF rate so we always show the newest frame (reduces lag from backlog).
  useEffect(() => {
    if (!isTauri || !isRunning) return;
    setCvFrame(null);
    latestCvFrameRef.current = null;
    setCvStreamError(false);
    let unlisten = null;
    let rafActive = true;
    const paintLatest = () => {
      if (!rafActive) return;
      const latest = latestCvFrameRef.current;
      if (latest != null) setCvFrame(latest);
      cvFrameRafRef.current = requestAnimationFrame(paintLatest);
    };
    cvFrameRafRef.current = requestAnimationFrame(paintLatest);
    Promise.all([
      import("@tauri-apps/api/core").then((m) => m.invoke("start_cv_feed")),
      import("@tauri-apps/api/event").then((m) =>
        m.listen("cv-frame", (ev) => {
          const b64 = ev.payload;
          if (typeof b64 === "string") latestCvFrameRef.current = `data:image/jpeg;base64,${b64}`;
        })
      ),
    ])
      .then(([, unlistenFn]) => {
        unlisten = unlistenFn;
        unlistenCvFrameRef.current = unlistenFn;
      })
      .catch((e) => {
        setCvStreamError(true);
        setError(String(e));
      });
    return () => {
      rafActive = false;
      if (cvFrameRafRef.current != null) cancelAnimationFrame(cvFrameRafRef.current);
      import("@tauri-apps/api/core").then((m) => m.invoke("stop_cv_feed")).catch(() => {});
      if (unlistenCvFrameRef.current) {
        unlistenCvFrameRef.current();
        unlistenCvFrameRef.current = null;
      }
      latestCvFrameRef.current = null;
      setCvFrame(null);
    };
  }, [isTauri, isRunning]);

  /** Start Workout: camera + skeleton from cv.py (native in Tauri, stream in browser). End Workout: show summary. */
  function handleToggle() {
    if (isRunning) {
      setError(null);
      setIsRunning(false);
      setShowSummary(true);
      return;
    }
    setError(null);
    setCvStreamError(false);
    sessionStartRef.current = Date.now();
    setLastRestAt(null);
    setTimeSinceRest(0);
    setStartRef.current = null;
    setShowSummary(false);
    setBetweenSetsMode(false);
    setIsRunning(true);
  }

  /** User starts next set after resting; CV will report reps for this set. */
  function startNextSet() {
    const now = Date.now();
    setBetweenSetsMode(false);
    setStartRef.current = now;
    setLastRestAt(now);
    setTimeSinceRest(0);
  }

  function handleRest() {
    const now = Date.now();
    setRestTimestamps((prev) => [...prev, now]);
    setLastRestAt(now);
    setTimeSinceRest(0);
  }

  /** Single path for rep events: button click now; CV rep detection / ML confirmation later. */
  function emitRepEvent() {
    const now = Date.now();
    if (setStartRef.current == null) setStartRef.current = now;
    setRepTimes((t) => [...t, now]);
  }

  /** CV calls this when bad form is detected; recorded per set and sent to backend on "Stop current set". */
  function recordFormIssue(issue) {
    setFormIssuesDuringSet((prev) => [...prev, issue]);
  }

  /** User stops current set: backend computes metrics; timer switches to Rest. Later: POST to backend. */
  function stopCurrentSet() {
    const setEndTime = Date.now();
    const setStartTime = setStartRef.current ?? sessionStartRef.current ?? setEndTime;
    const payload = {
      workoutId,
      setStartTime,
      setEndTime,
      repTimes: [...repTimes],
      formIssues: [...formIssuesDuringSet],
    };
    const metrics = commitSetStub(payload);
    setLastSetMetrics(metrics);
    setCompletedSets((prev) => [...prev, metrics]);
    setRestTimestamps((prev) => [...prev, setEndTime]);
    setLastRestAt(setEndTime);
    setTimeSinceRest(0);
    setRepTimes([]);
    setFormIssuesDuringSet([]);
    setStartRef.current = null;
    setBetweenSetsMode(true);
  }

  const avgTimeBetweenRest =
    restTimestamps.length >= 2
      ? restTimestamps
          .slice(1)
          .map((t, i) => (t - restTimestamps[i]) / 1000)
          .reduce((a, b) => a + b, 0) / (restTimestamps.length - 1)
      : null;

  const workoutConfig = workoutId ? WORKOUT_METRICS[workoutId] : null;
  const cornerRadius = 6;
  const totalReps = completedSets.reduce((sum, s) => sum + s.reps, 0);
  const betweenSets = isRunning && betweenSetsMode;

  // Unknown workout id: show message instead of full UI to avoid blank or broken screen.
  if (workoutId && !workoutConfig) {
    return (
      <Box sx={{ maxWidth: 1400, mx: "auto" }}>
        <Grow in timeout={400}>
          <Paper elevation={0} sx={{ p: 6, borderRadius: 4, textAlign: "center", background: "#ffffff", border: "1px solid #e5e7eb" }}>
            <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>Unknown workout</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>"{workoutId}" is not configured.</Typography>
            <Button variant="contained" size="large" onClick={() => navigate("/")} sx={{ px: 4, py: 1.5, borderRadius: 2, textTransform: "none", fontSize: "1rem", fontWeight: 600 }}>
              Choose Workout
            </Button>
          </Paper>
        </Grow>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1400, mx: "auto" }}>
      <Fade in timeout={300}>
        <Box sx={{ mb: 4 }}>
          <Typography 
            variant="h4" 
            sx={{ 
              fontWeight: 700,
              mb: 1,
              background: workoutConfig?.gradient || "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Live Session
          </Typography>
          {workoutConfig && (
            <Typography variant="body1" sx={{ color: "#6b7280" }}>
              {workoutConfig.title} Training
            </Typography>
          )}
        </Box>
      </Fade>

      {/* Summary shown after "End Workout"; Done dismisses and resets for next workout */}
      {showSummary && (
        <Grow in timeout={500}>
          <Paper
            elevation={0}
            sx={{
              p: 4,
              borderRadius: 4,
              background: "#ffffff",
              border: "1px solid #e5e7eb",
              position: "relative",
              overflow: "hidden",
              "&::before": {
                content: '""',
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: 6,
                background: workoutConfig?.gradient || "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              },
            }}
          >
            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
              <Box
                sx={{
                  width: 56,
                  height: 56,
                  borderRadius: 3,
                  background: workoutConfig?.gradient || "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <CheckCircleIcon sx={{ fontSize: 32, color: "#ffffff" }} />
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 600, mb: 0.5 }}>
                  Workout Complete!
                </Typography>
                {workoutConfig && (
                  <Typography variant="body2" color="text.secondary">
                    {workoutConfig.title}
                  </Typography>
                )}
              </Box>
            </Stack>

            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={6}>
                <Box sx={{ p: 2, borderRadius: 2, bgcolor: "#f9fafb" }}>
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: "block" }}>
                    Total Sets
                  </Typography>
                  <Typography variant="h4" sx={{ fontWeight: 700 }}>
                    {completedSets.length}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Box sx={{ p: 2, borderRadius: 2, bgcolor: "#f9fafb" }}>
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: "block" }}>
                    Total Reps
                  </Typography>
                  <Typography variant="h4" sx={{ fontWeight: 700 }}>
                    {totalReps}
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            <Stack spacing={1.5} sx={{ mb: 3 }}>
              {completedSets.map((set, i) => (
                <Fade in key={i} timeout={300 + i * 100}>
                  <Box
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      border: "1px solid #e5e7eb",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Set {i + 1}
                    </Typography>
                    <Stack direction="row" spacing={3}>
                      <Typography variant="body2">
                        <strong>{set.reps}</strong> reps
                      </Typography>
                      {set.avgSec != null && (
                        <Typography variant="body2" color="text.secondary">
                          {set.avgSec}s/rep avg
                        </Typography>
                      )}
                    </Stack>
                  </Box>
                </Fade>
              ))}
            </Stack>

            <Button
              fullWidth
              variant="contained"
              size="large"
              onClick={() => {
                setShowSummary(false);
                setCompletedSets([]);
                setLastSetMetrics(null);
                setRestTimestamps([]);
                setLastRestAt(null);
              }}
              sx={{
                py: 1.5,
                borderRadius: 2,
                background: workoutConfig?.gradient || "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                textTransform: "none",
                fontSize: "1rem",
                fontWeight: 600,
                "&:hover": {
                  opacity: 0.9,
                },
              }}
            >
              Done
            </Button>
          </Paper>
        </Grow>
      )}

      {!showSummary && !workoutId && (
        <Grow in timeout={400}>
          <Paper
            elevation={0}
            sx={{
              p: 6,
              borderRadius: 4,
              textAlign: "center",
              background: "#ffffff",
              border: "1px solid #e5e7eb",
            }}
          >
            <FitnessCenterIcon sx={{ fontSize: 64, color: "#d1d5db", mb: 2 }} />
            <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
              No Workout Selected
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Choose a workout to start your training session
            </Typography>
            <Button 
              variant="contained" 
              size="large"
              onClick={() => navigate("/")}
              sx={{
                px: 4,
                py: 1.5,
                borderRadius: 2,
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                textTransform: "none",
                fontSize: "1rem",
                fontWeight: 600,
              }}
            >
              Choose Workout
            </Button>
          </Paper>
        </Grow>
      )}

      {!showSummary && workoutId && (
        <>
      {/* Camera Feed - Full Width */}
      <Grow in timeout={400}>
        <Paper
          elevation={0}
          sx={{
            width: "100%",
            maxWidth: 960,
            mx: "auto",
            borderRadius: 4,
            overflow: "hidden",
            background: "#1a1a1a",
            position: "relative",
            aspectRatio: "16 / 9",
            border: "1px solid #2d2d2d",
          }}
        >
          {isRunning && isTauri && cvFrame && (
            <img
              key="cv-native"
              src={cvFrame}
              alt="Camera with skeleton overlay (cv.py)"
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                objectFit: "contain",
              }}
            />
          )}
          {isRunning && isTauri && !cvFrame && !cvStreamError && (
            <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#1a1a1a" }}>
              <Typography sx={{ color: "#9ca3af" }}>Starting cv.py…</Typography>
            </Box>
          )}
          {isRunning && isTauri && cvStreamError && (
            <Box
              sx={{
                position: "absolute",
                inset: 0,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 2,
                p: 2,
                background: "#1a1a1a",
              }}
            >
              <Typography sx={{ color: "#fbbf24", fontSize: "0.875rem", textAlign: "center" }}>
                cv.py failed to start. Check Python and cv/config.yaml (camera_id). Run from repo: npm run tauri dev.
              </Typography>
            </Box>
          )}
          {isRunning && !isTauri && !cvStreamError && (
            <img
              key="cv-stream"
              src={CV_STREAM_URL}
              alt="Camera with skeleton overlay (cv.py stream)"
              onError={() => setCvStreamError(true)}
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                objectFit: "contain",
              }}
            />
          )}
          {isRunning && !isTauri && cvStreamError && (
            <Box
              sx={{
                position: "absolute",
                inset: 0,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 2,
                p: 2,
                background: "#1a1a1a",
              }}
            >
              <Typography sx={{ color: "#fbbf24", fontSize: "0.875rem", textAlign: "center" }}>
                Camera and skeleton use cv.py only. Run: python cv/cv_stream_server.py (camera from cv/config.yaml).
              </Typography>
            </Box>
          )}
          {!isRunning && (
            <>
              <Box
                sx={{
                  position: "absolute",
                  inset: 24,
                  borderRadius: 3,
                  border: "2px dashed #3d3d3d",
                }}
              />
              <Box
                sx={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexDirection: "column",
                  color: "#9ca3af",
                  gap: 1,
                }}
              >
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {error ? error : "Camera + skeleton (cv.py)"}
                </Typography>
                <Typography variant="body2">Start workout to see feed from cv.py. Run: python cv/cv_stream_server.py</Typography>
              </Box>
            </>
          )}
          <Box sx={{ position: "absolute", left: 20, top: 20 }}>
            <Chip
              icon={isRunning && !cvStreamError && (isTauri ? cvFrame : true) ? <CheckCircleIcon /> : undefined}
              label={
                !isRunning ? "Camera Off" :
                cvStreamError ? "CV unavailable" : isTauri ? "Live (cv.py native)" : "Live (cv.py)"
              }
              sx={{
                background: isRunning && !cvStreamError
                  ? "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                  : "#374151",
                color: "#ffffff",
                fontWeight: 600,
                px: 1,
              }}
            />
          </Box>
        </Paper>
      </Grow>

      {/* Metrics Grid */}
      <Grid container spacing={2} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6} lg={4}>
          {/* Current Set Metrics */}
            <Grow in timeout={500}>
              <Paper
                elevation={0}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  background: "#ffffff",
                  border: "1px solid #e5e7eb",
                  position: "relative",
                  overflow: "hidden",
                  height: "100%",
                  "&::before": {
                    content: '""',
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: 4,
                    background: workoutConfig?.gradient,
                  },
                }}
              >
                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2 }}>
                  <Box
                    sx={{
                      width: 36,
                      height: 36,
                      borderRadius: 2,
                      background: workoutConfig?.gradient,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <FitnessCenterIcon sx={{ fontSize: 18, color: "#ffffff" }} />
                  </Box>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {workoutConfig?.title}
                  </Typography>
                </Stack>

                {isRunning && (
                  <Fade in timeout={300}>
                    <Box
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        background: "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
                        mb: 1.5,
                        textAlign: "center",
                      }}
                    >
                      <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: "block" }}>
                        Current Set
                      </Typography>
                      <Typography variant="h2" sx={{ fontWeight: 700, color: "#059669" }}>
                        {repTimes.length}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        reps
                      </Typography>
                    </Box>
                  </Fade>
                )}

                {lastSetMetrics != null && (
                  <Fade in timeout={300}>
                    <Stack spacing={1} sx={{ mb: 1.5 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, fontSize: "0.7rem" }}>
                        LAST SET
                      </Typography>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.85rem" }}>
                          Total Reps
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {lastSetMetrics.reps}
                        </Typography>
                      </Box>
                      {lastSetMetrics.avgSec != null && (
                        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.85rem" }}>
                            Avg Time
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {lastSetMetrics.avgSec}s
                          </Typography>
                        </Box>
                      )}
                    </Stack>
                  </Fade>
                )}

                <Typography variant="caption" color="text.secondary" sx={{ mb: 1.5, display: "block", fontSize: "0.7rem" }}>
                  {workoutConfig?.extra}
                </Typography>

                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={emitRepEvent}
                  disabled={!isRunning}
                  sx={{
                    py: 1.2,
                    borderRadius: 2,
                    borderColor: "#e5e7eb",
                    textTransform: "none",
                    fontWeight: 600,
                    fontSize: "0.9rem",
                    "&:hover": {
                      borderColor: "#9ca3af",
                      background: "#f9fafb",
                    },
                    "&:disabled": {
                      borderColor: "#e5e7eb",
                      color: "#9ca3af",
                    },
                  }}
                >
                  Add Rep
                </Button>
              </Paper>
            </Grow>

        </Grid>

        <Grid item xs={12} md={6} lg={4}>
          {/* Rest Timer */}
          <Grow in timeout={600}>
              <Paper
                elevation={0}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  background: betweenSetsMode 
                    ? "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
                    : "#ffffff",
                  border: "1px solid #e5e7eb",
                  transition: "all 0.3s ease",
                  height: "100%",
                }}
              >
                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2 }}>
                  <Box
                    sx={{
                      width: 36,
                      height: 36,
                      borderRadius: 2,
                      background: betweenSetsMode
                        ? "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
                        : "#f3f4f6",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      transition: "all 0.3s ease",
                    }}
                  >
                    <TimerIcon sx={{ fontSize: 18, color: betweenSetsMode ? "#ffffff" : "#6b7280" }} />
                  </Box>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {betweenSetsMode ? "Rest Time" : "Active"}
                  </Typography>
                </Stack>

                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    background: "#ffffff",
                    textAlign: "center",
                    mb: 1.5,
                  }}
                >
                  <Typography variant="h2" sx={{ fontWeight: 700, color: betweenSetsMode ? "#d97706" : "#6b7280" }}>
                    {timeSinceRest}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    seconds
                  </Typography>
                </Box>

                {avgTimeBetweenRest != null && (
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1.5, display: "block", textAlign: "center", fontSize: "0.7rem" }}>
                    Avg: {avgTimeBetweenRest.toFixed(1)}s
                  </Typography>
                )}

                <Button
                  fullWidth
                  variant="outlined"
                  onClick={handleRest}
                  disabled={!isRunning}
                  sx={{
                    py: 1.2,
                    borderRadius: 2,
                    borderColor: "#e5e7eb",
                    textTransform: "none",
                    fontWeight: 600,
                    fontSize: "0.9rem",
                    "&:hover": {
                      borderColor: "#9ca3af",
                      background: "#f9fafb",
                    },
                  }}
                >
                  Mark Rest
                </Button>
              </Paper>
            </Grow>
        </Grid>

        <Grid item xs={12} md={12} lg={4}>
          {/* Feedback Section */}
          <Grow in timeout={700}>
        <Paper
          elevation={0}
          sx={{
            p: 2.5,
            borderRadius: 3,
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            height: "100%",
          }}
        >
          <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2 }}>
            <Box
              sx={{
                width: 36,
                height: 36,
                borderRadius: 2,
                background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <FeedbackIcon sx={{ fontSize: 18, color: "#ffffff" }} />
            </Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Form Feedback
            </Typography>
          </Stack>

          <Box
            sx={{
              p: 2,
              borderRadius: 2,
              background: "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
              border: "1px solid #86efac",
            }}
          >
            <Typography variant="body2" sx={{ color: "#065f46", fontWeight: 500 }}>
              Good depth. Keep knees aligned.
            </Typography>
          </Box>
        </Paper>
      </Grow>
        </Grid>
      </Grid>

      {/* Action Buttons */}
      <Fade in timeout={800}>
        <Stack 
          direction="row" 
          spacing={2} 
          sx={{ 
            mt: 3, 
            justifyContent: "center",
            flexWrap: "wrap",
            gap: 2,
          }}
        >
          <Button
            variant={isRunning ? "outlined" : "contained"}
            size="large"
            startIcon={isRunning ? <StopIcon /> : <PlayArrowIcon />}
            onClick={handleToggle}
            sx={{
              px: 3.5,
              py: 1.2,
              borderRadius: 2,
              textTransform: "none",
              fontWeight: 600,
              fontSize: "0.95rem",
              minWidth: 160,
              ...(!isRunning && {
                background: workoutConfig?.gradient || "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              }),
              ...(isRunning && {
                borderColor: "#ef4444",
                color: "#ef4444",
                "&:hover": {
                  borderColor: "#dc2626",
                  background: "#fef2f2",
                },
              }),
            }}
          >
            {isRunning ? "End Workout" : "Start Workout"}
          </Button>

          {betweenSets && (
            <Slide in direction="right" timeout={300}>
              <Button 
                variant="contained" 
                size="large"
                onClick={startNextSet}
                sx={{
                  px: 3.5,
                  py: 1.2,
                  borderRadius: 2,
                  textTransform: "none",
                  fontWeight: 600,
                  fontSize: "0.95rem",
                  minWidth: 160,
                  background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                }}
              >
                Start Next Set
              </Button>
            </Slide>
          )}

          <Button
            variant="outlined"
            size="large"
            onClick={stopCurrentSet}
            disabled={!isRunning || repTimes.length === 0}
            sx={{
              px: 3.5,
              py: 1.2,
              borderRadius: 2,
              textTransform: "none",
              fontWeight: 600,
              fontSize: "0.95rem",
              minWidth: 160,
              borderColor: "#e5e7eb",
              "&:hover": {
                borderColor: "#9ca3af",
                background: "#f9fafb",
              },
            }}
          >
            Stop Current Set
          </Button>
        </Stack>
      </Fade>
        </>
      )}
    </Box>
  );
}
