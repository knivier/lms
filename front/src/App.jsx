import {
  AppBar,
  BottomNavigation,
  BottomNavigationAction,
  Box,
  Fade,
  Toolbar,
  Typography,
} from "@mui/material";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import VideocamOutlinedIcon from "@mui/icons-material/VideocamOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import { Routes, Route, useLocation, useNavigate } from "react-router-dom";
import WorkoutPicker from "./pages/WorkoutPicker";
import LiveSession from "./pages/LiveSession";
import Settings from "./pages/Settings";

const titlesByPath = {
  "/": "Workout Picker",
  "/live": "Live Session",
  "/settings": "Settings",
};

function App() {
  const location = useLocation();
  const navigate = useNavigate();

  const currentPath =
    location.pathname === "/" ? "/" : location.pathname;

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        bgcolor: "background.default",
      }}
    >
      <AppBar position="sticky" color="transparent" elevation={0}>
        <Toolbar sx={{ py: 1.5 }}>
          <Typography variant="h6">
            {titlesByPath[currentPath] ?? "Fitness"}
          </Typography>
        </Toolbar>
      </AppBar>

      <Box sx={{ flex: 1, px: 4, py: 4 }}>
        <Fade in key={location.pathname} timeout={300}>
          <Box>
            <Routes location={location}>
              <Route path="/" element={<WorkoutPicker />} />
              <Route path="/live" element={<LiveSession />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Box>
        </Fade>
      </Box>

      <Box sx={{ px: 2, pb: 2 }}>
        <BottomNavigation
          value={currentPath}
          onChange={(_, value) => navigate(value)}
          showLabels
          sx={{ height: 72, borderRadius: 4, bgcolor: "background.paper" }}
        >
          <BottomNavigationAction
            label="Workouts"
            value="/"
            icon={<FitnessCenterIcon />}
          />
          <BottomNavigationAction
            label="Live"
            value="/live"
            icon={<VideocamOutlinedIcon />}
          />
          <BottomNavigationAction
            label="Settings"
            value="/settings"
            icon={<SettingsOutlinedIcon />}
          />
        </BottomNavigation>
      </Box>
    </Box>
  );
}

export default App;
