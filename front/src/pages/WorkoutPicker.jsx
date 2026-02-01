import { Box, Stack, Typography, Fade, Grow } from "@mui/material";
import { useNavigate } from "react-router-dom";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import SportsGymnasticsIcon from "@mui/icons-material/SportsGymnastics";
import { useState } from "react";

export const WORKOUTS = [
  { 
    id: "squat", 
    title: "Squat", 
    description: "Lower body strength and control.",
    icon: DirectionsRunIcon,
    gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "#667eea"
  },
  { 
    id: "pushup", 
    title: "Push-up", 
    description: "Upper body endurance and form.",
    icon: SportsGymnasticsIcon,
    gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    color: "#f093fb"
  },
  { 
    id: "bicep_curl", 
    title: "Bicep Curl", 
    description: "Arm strength and tempo.",
    icon: FitnessCenterIcon,
    gradient: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    color: "#4facfe"
  },
];

function WorkoutCard({ workout, index, onClick }) {
  const [isHovered, setIsHovered] = useState(false);
  const Icon = workout.icon;

  return (
    <Grow in timeout={400 + index * 150}>
      <Box
        onClick={onClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        sx={{
          position: "relative",
          cursor: "pointer",
          borderRadius: 4,
          overflow: "hidden",
          background: "#ffffff",
          boxShadow: isHovered 
            ? "0 20px 40px rgba(0,0,0,0.12)" 
            : "0 4px 12px rgba(0,0,0,0.08)",
          transform: isHovered ? "translateY(-8px) scale(1.02)" : "translateY(0) scale(1)",
          transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
          "&::before": {
            content: '""',
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "6px",
            background: workout.gradient,
            transform: isHovered ? "scaleX(1)" : "scaleX(0)",
            transformOrigin: "left",
            transition: "transform 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
          },
        }}
      >
        <Box sx={{ p: 4, display: "flex", alignItems: "center", gap: 3 }}>
          <Box
            sx={{
              width: 72,
              height: 72,
              borderRadius: 3,
              background: workout.gradient,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transform: isHovered ? "rotate(5deg) scale(1.1)" : "rotate(0deg) scale(1)",
              transition: "transform 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
              boxShadow: isHovered 
                ? `0 8px 24px ${workout.color}40`
                : `0 4px 12px ${workout.color}30`,
            }}
          >
            <Icon sx={{ fontSize: 36, color: "#ffffff" }} />
          </Box>
          
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography 
              variant="h5" 
              sx={{ 
                fontWeight: 600,
                mb: 0.5,
                color: "#1a1a1a",
                transition: "color 0.3s ease",
              }}
            >
              {workout.title}
            </Typography>
            <Typography 
              variant="body2" 
              sx={{ 
                color: "#6b7280",
                lineHeight: 1.6,
              }}
            >
              {workout.description}
            </Typography>
          </Box>

          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: "50%",
              background: isHovered ? workout.gradient : "#f3f4f6",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
              transform: isHovered ? "translateX(4px)" : "translateX(0)",
            }}
          >
            <Box
              component="span"
              sx={{
                width: 0,
                height: 0,
                borderLeft: isHovered ? "8px solid #ffffff" : "8px solid #9ca3af",
                borderTop: "5px solid transparent",
                borderBottom: "5px solid transparent",
                ml: "2px",
                transition: "border-color 0.3s ease",
              }}
            />
          </Box>
        </Box>
      </Box>
    </Grow>
  );
}

export default function WorkoutPicker() {
  const navigate = useNavigate();

  async function handleSelectWorkout(workoutId) {
    if (typeof window !== "undefined" && (window.__TAURI_INTERNALS__ != null || window.__TAURI__ != null)) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("write_workout_id", { workoutId });
      } catch (_) {}
    }
    navigate("/live", { state: { workoutId } });
  }

  return (
    <Box sx={{ maxWidth: 800, mx: "auto" }}>
      <Fade in timeout={300}>
        <Box>
          <Typography 
            variant="h4" 
            sx={{ 
              fontWeight: 700,
              mb: 1,
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Choose Workout
          </Typography>
          <Typography 
            variant="body1" 
            sx={{ 
              color: "#6b7280",
              mb: 4,
            }}
          >
            Select an exercise to begin your training session
          </Typography>
        </Box>
      </Fade>

      <Stack spacing={3}>
        {WORKOUTS.map((workout, index) => (
          <WorkoutCard
            key={workout.id}
            workout={workout}
            index={index}
            onClick={() => handleSelectWorkout(workout.id)}
          />
        ))}
      </Stack>
    </Box>
  );
}
