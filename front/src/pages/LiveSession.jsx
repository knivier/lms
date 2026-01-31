import { Box, Button, Chip, Stack, Typography } from "@mui/material";
import { useState, useRef, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

// Metrics config per workout (title + exercise-specific metric placeholders)
const WORKOUT_METRICS = {
  squat: { title: "Squat", extra: "Depth: — · Knees: —" },
  pushup: { title: "Push-up", extra: "Chest touch: — · Lockout: —" },
  bicep_curl: { title: "Bicep curl", extra: "ROM: — · Elbow drift: —" },
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
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);

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

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }
  }, [stream]);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [stream]);

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

  /** Start Workout: system initializes, camera on; CV reports reps. End Workout: stop camera and show summary. */
  async function handleToggle() {
    if (isRunning) {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        setStream(null);
      }
      setError(null);
      setIsRunning(false);
      setShowSummary(true);
      return;
    }
    setError(null);
    sessionStartRef.current = Date.now();
    setLastRestAt(null);
    setTimeSinceRest(0);
    setStartRef.current = null;
    setShowSummary(false);
    setBetweenSetsMode(false);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1200, height: 768 },
      });
      setStream(mediaStream);
      setIsRunning(true);
    } catch (e) {
      setError(e.message || "Could not access camera");
    }
  }

  /** User starts next set after resting; CV will report reps for this set. */
  function startNextSet() {
    setBetweenSetsMode(false);
    setStartRef.current = Date.now();
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

  return (
    <Stack spacing={3}>
      <Typography variant="h4">Live Session</Typography>

      {/* Summary shown after "End Workout"; Done dismisses and resets for next workout */}
      {showSummary && (
        <Box
          sx={{
            p: 3,
            borderRadius: 2,
            bgcolor: "primary.50",
            border: "1px solid",
            borderColor: "primary.200",
          }}
        >
          <Typography variant="h6" gutterBottom>
            Workout Summary
          </Typography>
          {workoutConfig && (
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {workoutConfig.title}
            </Typography>
          )}
          <Stack spacing={0.5} sx={{ mb: 2 }}>
            {completedSets.map((set, i) => (
              <Typography key={i} variant="body2">
                Set {i + 1}: <strong>{set.reps}</strong> reps
                {set.avgSec != null && ` · avg ${set.avgSec}s/rep`}
              </Typography>
            ))}
          </Stack>
          <Typography variant="body2" fontWeight={600}>
            Total reps: {totalReps}
          </Typography>
          <Button
            variant="contained"
            onClick={() => {
              setShowSummary(false);
              setCompletedSets([]);
              setLastSetMetrics(null);
              setRestTimestamps([]);
              setLastRestAt(null);
            }}
            sx={{ mt: 2 }}
          >
            Done
          </Button>
        </Box>
      )}

      {!showSummary && !workoutId && (
        <Box
          sx={{
            p: 3,
            borderRadius: 2,
            bgcolor: "grey.100",
            border: "1px solid",
            borderColor: "grey.300",
          }}
        >
          <Typography variant="body1" color="text.secondary" gutterBottom>
            Select a workout from the Workouts tab to see metrics here.
          </Typography>
          <Button variant="contained" onClick={() => navigate("/")}>
            Choose workout
          </Button>
        </Box>
      )}

      {!showSummary && workoutId && (
        <>
      <Box
        sx={{
          width: "100%",
          aspectRatio: "16 / 9",
          maxWidth: 720,
          borderRadius: cornerRadius,
          bgcolor: "grey.900",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {stream && (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              borderRadius: cornerRadius,
            }}
          />
        )}
        {!stream && (
          <>
            <Box
              sx={{
                position: "absolute",
                inset: 24,
                borderRadius: cornerRadius,
                border: "2px dashed",
                borderColor: "grey.700",
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
                color: "grey.500",
                gap: 0.5,
              }}
            >
              <Typography variant="subtitle1">
                {error ? error : "Camera feed"}
              </Typography>
              <Typography variant="caption">Skeleton overlay</Typography>
            </Box>
          </>
        )}
        <Box sx={{ position: "absolute", left: 24, top: 24 }}>
          <Chip
            label={stream ? "Locked on" : "Camera off"}
            color={stream ? "primary" : "default"}
            size="small"
          />
        </Box>
      </Box>

      {/* Metrics for the selected workout only. Current set = live (CV or + Rep); last set = backend after "Stop current set". */}
      {workoutConfig && (
        <Box
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: "grey.100",
            border: "1px solid",
            borderColor: "grey.300",
            maxWidth: 400,
          }}
        >
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
            {workoutConfig.title}
          </Typography>
          <Stack spacing={0.5}>
            {isRunning && (
              <Typography variant="body2">
                Current set: <strong>{repTimes.length}</strong> reps
              </Typography>
            )}
            {lastSetMetrics != null && (
              <>
                <Typography variant="body2" color="text.secondary">
                  Last set: <strong>{lastSetMetrics.reps}</strong> reps
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Time for last rep: {lastSetMetrics.lastRepSec != null ? `${lastSetMetrics.lastRepSec}s` : "—"}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg time/rep: {lastSetMetrics.avgSec != null ? `${lastSetMetrics.avgSec}s` : "—"}
                </Typography>
              </>
            )}
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
              {workoutConfig.extra}
            </Typography>
            <Button
              size="small"
              variant="outlined"
              onClick={emitRepEvent}
              disabled={!isRunning}
            >
              + Rep
            </Button>
          </Stack>
        </Box>
      )}

      {/* Rest: when between sets show "Rest time"; when in a set show "Time since last rest" */}
      <Stack spacing={0.5}>
        <Typography variant="h6">Rest</Typography>
        <Typography variant="body2" color="text.secondary">
          {betweenSetsMode ? "Rest time" : "Time since last rest"}: {timeSinceRest}s
        </Typography>
        {avgTimeBetweenRest != null && (
          <Typography variant="body2" color="text.secondary">
            Avg time between rest: {avgTimeBetweenRest.toFixed(1)}s
          </Typography>
        )}
        <Button variant="outlined" size="small" onClick={handleRest} disabled={!isRunning}>
          Mark rest
        </Button>
      </Stack>

      {/* Feedback: later use structured data { severity, message, joint } for color, animation, per-joint explanation */}
      <Stack spacing={1}>
        <Typography variant="h6">Feedback</Typography>
        <Typography variant="body1" color="text.secondary">
          Good depth. Keep knees aligned.
        </Typography>
      </Stack>

      <Stack direction="row" spacing={2} flexWrap="wrap">
        <Button
          variant={isRunning ? "outlined" : "contained"}
          onClick={handleToggle}
        >
          {isRunning ? "End Workout" : "Start Workout"}
        </Button>
        {betweenSets && (
          <Button variant="contained" onClick={startNextSet}>
            Start next set
          </Button>
        )}
        <Button
          variant="outlined"
          onClick={stopCurrentSet}
          disabled={!isRunning || repTimes.length === 0}
        >
          Stop current set
        </Button>
      </Stack>
        </>
      )}
    </Stack>
  );
}
